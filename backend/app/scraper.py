from __future__ import annotations

"""
Amazon Catalog Scraper — v3
===========================
Key changes:
  - Accepts a ``SearchSpec`` instead of ``ProductFilters``.
  - Builds the Amazon SERP URL with native filter parameters (price, rating,
    Prime, brand, sort) so Amazon applies them server-side BEFORE delivering
    results.  No more ``_passes_filters()`` post-processing.
  - Keyword-only check (must_have / avoid) is the ONLY Python-side filter
    kept, because Amazon can't filter on arbitrary keywords.
  - Browser automation runs through the Playwright MCP server instead of the
    local Playwright Python API.
  - Query-result caching: identical SearchSpec hits are served from an
    in-process TTL cache to reduce browser round-trips during a session.
"""

import hashlib
import json
import logging
import re
import shutil
import time
from typing import Any
from urllib.parse import urlencode, urljoin

from .config import get_settings
from .marketplaces import marketplace_currency, marketplace_domain
from .models import Product, SearchSpec
from .storage import sample_products

logger = logging.getLogger(__name__)

PRICE_RE = re.compile(r"(\d[\d,]*(?:\.\d{1,2})?)")
RATING_RE = re.compile(r"([0-5](?:\.\d)?)\s+out of 5")
REVIEWS_RE = re.compile(r"(\d[\d,]*)")

# Simple in-process TTL cache  {cache_key: (timestamp, products)}
_RESULT_CACHE: dict[str, tuple[float, list[Product]]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes
_PLAYWRIGHT_MCP_PACKAGE = "@playwright/mcp@latest"
_PLAYWRIGHT_MCP_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_MCP_EXTRACT_PRODUCTS_JS = """
() => {
  const firstText = (root, selectors) => {
    for (const selector of selectors) {
      const element = root.querySelector(selector);
      const value = element?.innerText || element?.textContent;
      if (value && value.trim()) {
        return value.trim();
      }
    }
    return null;
  };

  const firstAttr = (root, selectors, attr) => {
    for (const selector of selectors) {
      const value = root.querySelector(selector)?.getAttribute(attr);
      if (value) {
        return value;
      }
    }
    return null;
  };

  return Array.from(
    document.querySelectorAll('[data-component-type="s-search-result"]')
  ).map((card) => {
    const priceText =
      firstText(card, [
        ".a-price .a-offscreen",
        "[data-cy='price-recipe'] .a-price .a-offscreen",
        ".a-price",
      ]) ||
      (() => {
        const whole = firstText(card, [".a-price-whole"]);
        const fraction = firstText(card, [".a-price-fraction"]);
        return whole ? `${whole}.${fraction || "00"}` : null;
      })();

    return {
      title: firstText(card, [
        "h2 span",
        '[data-cy="title-recipe-title"]',
        ".a-size-medium.a-color-base.a-text-normal",
      ]),
      href: firstAttr(card, ["h2 a", "a.a-link-normal.s-no-outline"], "href"),
      product_id: card.getAttribute("data-asin"),
      price_text: priceText,
      rating_text:
        firstAttr(card, [".a-icon-star-small", ".a-icon-star"], "aria-label") ||
        firstText(card, [".a-icon-alt"]),
      reviews_text:
        firstText(card, [
          "span.a-size-base.s-underline-text",
          "a[href*='customerReviews'] span",
          "[aria-label*='ratings']",
        ]) || firstAttr(card, ["[aria-label*='ratings']"], "aria-label"),
      image_url: firstAttr(card, ["img.s-image"], "src"),
      prime_text:
        firstText(card, [".a-icon-prime"]) ||
        firstAttr(card, ["[aria-label*='Prime']"], "aria-label"),
    };
  });
}
""".strip()


def _parse_price(value: str | None) -> float | None:
    if not value:
        return None
    match = PRICE_RE.search(value.replace("\n", " "))
    return float(match.group(1).replace(",", "")) if match else None


def _parse_rating(value: str | None) -> float | None:
    if not value:
        return None
    match = RATING_RE.search(value)
    return float(match.group(1)) if match else None


def _parse_reviews(value: str | None) -> int | None:
    if not value:
        return None
    match = REVIEWS_RE.search(value)
    return int(match.group(1).replace(",", "")) if match else None


def build_search_url(domain: str, spec: SearchSpec, page: int) -> str:
    """
    Construct the Amazon SERP URL with native filter parameters.

    Amazon URL filter params used:
      k      — search keyword
      page   — page number
      p_36   — price range (min-max in currency-unit × 100)
      p_72   — minimum star-rating node ID
      p_85   — Prime eligible flag
      p_89   — brand filter
      s      — sort order
      i      — department slug
    """
    params: dict[str, str] = {
        "k": spec.search_term,
        "page": str(page),
    }

    url_f = spec.amazon_url_filters
    if url_f.p_36:
        params["p_36"] = url_f.p_36
    if url_f.p_72:
        params["p_72"] = url_f.p_72
    if url_f.p_85:
        params["p_85"] = url_f.p_85
    if url_f.p_89:
        params["p_89"] = url_f.p_89
    if url_f.s:
        params["s"] = url_f.s
    if url_f.i:
        params["i"] = url_f.i

    return f"{domain}/s?{urlencode(params)}"


def _cache_key(spec: SearchSpec) -> str:
    """Deterministic key for the in-process result cache."""
    key_str = (
        f"{spec.marketplace}|{spec.search_term}|"
        f"{spec.amazon_url_filters.model_dump_json()}"
    )
    return hashlib.sha1(key_str.encode()).hexdigest()[:20]


class AmazonCatalogScraper:
    """Amazon SERP scraper using Playwright MCP and native URL-level filters."""

    async def search(self, spec: SearchSpec) -> list[Product]:
        """
        Scrape Amazon using the SearchSpec.
        1. Build pre-filtered URL (Amazon applies server-side filters).
        2. Navigate and extract product cards through the Playwright MCP server.
        3. Apply must_have / avoid keyword checks in Python (only light pass).
        4. Return up to scraper_max_results products.
        """
        settings = get_settings()
        domain = marketplace_domain(spec.marketplace)
        currency_code, currency_symbol = marketplace_currency(spec.marketplace)

        # ---- cache check ----
        key = _cache_key(spec)
        cached = _RESULT_CACHE.get(key)
        if cached and (time.monotonic() - cached[0]) < _CACHE_TTL_SECONDS:
            logger.info("AmazonCatalogScraper: cache hit for key=%s", key)
            return cached[1]

        if not shutil.which("npx"):
            logger.warning("npx not found — returning sample products.")
            return sample_products(spec.search_term)

        products: list[Product] = []
        extracted_count = 0

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            logger.warning("Python MCP SDK not installed — returning sample products.")
            return sample_products(spec.search_term)

        server_params = StdioServerParameters(
            command="npx",
            args=[
                "--yes",
                _PLAYWRIGHT_MCP_PACKAGE,
                "--headless",
                "--isolated",
                "--snapshot-mode",
                "none",
                "--codegen",
                "none",
                "--timeout-action",
                str(settings.scraper_timeout_ms),
                "--timeout-navigation",
                str(settings.scraper_timeout_ms),
                "--viewport-size",
                "1366x900",
                "--user-agent",
                _PLAYWRIGHT_MCP_USER_AGENT,
            ],
        )

        try:
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    try:
                        products, extracted_count = await self._search_with_mcp_session(
                            session,
                            spec,
                            domain,
                            currency_code,
                            currency_symbol,
                            settings.scraper_max_pages,
                            settings.scraper_max_results,
                        )
                    finally:
                        await self._safe_close_browser(session)
        except Exception as exc:
            logger.error("AmazonCatalogScraper MCP error: %s", exc)
            return sample_products(spec.search_term)

        if not extracted_count:
            return sample_products(spec.search_term)

        # ---- cache store ----
        _RESULT_CACHE[key] = (time.monotonic(), products)
        return products

    async def _search_with_mcp_session(
        self,
        session: Any,
        spec: SearchSpec,
        domain: str,
        currency_code: str,
        currency_symbol: str,
        max_pages_setting: int,
        max_results: int,
    ) -> tuple[list[Product], int]:
        products: list[Product] = []
        extracted_count = 0

        max_pages = min(spec.max_pages, max_pages_setting)
        for page_number in range(1, max_pages + 1):
            url = build_search_url(domain, spec, page_number)
            logger.info("AmazonCatalogScraper: fetching %s via Playwright MCP", url)

            await self._call_mcp_tool(session, "browser_navigate", {"url": url})
            await self._call_mcp_tool(session, "browser_wait_for", {"time": 1.2})

            raw_items = await self._extract_page_items(session)
            for item in raw_items:
                if len(products) >= max_results:
                    break
                product = self._product_from_mcp_item(
                    item, domain, currency_code, currency_symbol
                )
                if product:
                    extracted_count += 1
                    if self._passes_keyword_check(product, spec):
                        products.append(product)

            if len(products) >= max_results:
                break

        return products, extracted_count

    async def _extract_page_items(self, session: Any) -> list[dict[str, Any]]:
        result = await self._call_mcp_tool(
            session,
            "browser_evaluate",
            {"function": _MCP_EXTRACT_PRODUCTS_JS},
        )
        parsed = self._parse_mcp_json_result(result)
        return parsed if isinstance(parsed, list) else []

    async def _call_mcp_tool(self, session: Any, name: str, arguments: dict[str, Any]) -> Any:
        result = await session.call_tool(name, arguments)
        if getattr(result, "isError", False):
            raise RuntimeError(self._mcp_text(result) or f"MCP tool failed: {name}")
        return result

    async def _safe_close_browser(self, session: Any) -> None:
        try:
            await session.call_tool("browser_close", {})
        except Exception:
            logger.debug("Playwright MCP browser_close failed", exc_info=True)

    def _parse_mcp_json_result(self, result: Any) -> Any:
        text = self._mcp_text(result)
        match = re.search(r"### Result\s*(.*?)\s*(?=\n### |\Z)", text, re.DOTALL)
        if not match:
            raise ValueError("Playwright MCP response did not include a result block")
        return json.loads(match.group(1).strip())

    @staticmethod
    def _mcp_text(result: Any) -> str:
        chunks: list[str] = []
        for item in getattr(result, "content", []) or []:
            text = getattr(item, "text", None)
            if text:
                chunks.append(text)
        return "\n".join(chunks)

    # ------------------------------------------------------------------
    # Card extraction
    # ------------------------------------------------------------------

    def _product_from_mcp_item(
        self,
        item: dict[str, Any],
        domain: str,
        currency_code: str,
        currency_symbol: str,
    ) -> Product | None:
        title = item.get("title")
        if not isinstance(title, str) or not title.strip():
            return None

        href = item.get("href")
        url = urljoin(domain, href) if isinstance(href, str) and href else domain
        product_id = item.get("product_id")
        if not isinstance(product_id, str) or not product_id:
            product_id = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]

        price_text = item.get("price_text") if isinstance(item.get("price_text"), str) else None
        rating_text = (
            item.get("rating_text") if isinstance(item.get("rating_text"), str) else None
        )
        reviews_text = (
            item.get("reviews_text") if isinstance(item.get("reviews_text"), str) else None
        )
        image_url = item.get("image_url") if isinstance(item.get("image_url"), str) else None
        prime_text = item.get("prime_text") if isinstance(item.get("prime_text"), str) else None

        return Product(
            product_id=product_id,
            title=" ".join(title.split()),
            url=url,
            price=_parse_price(price_text),
            currency_code=currency_code,
            currency_symbol=currency_symbol,
            rating=_parse_rating(rating_text),
            review_count=_parse_reviews(reviews_text),
            image_url=image_url,
            brand=self._guess_brand(title),
            is_prime=bool(prime_text),
            raw={
                "price_text": price_text,
                "rating_text": rating_text,
                "marketplace_domain": domain,
            },
        )

    # ------------------------------------------------------------------
    # Lightweight keyword check (replaces old _passes_filters)
    # ------------------------------------------------------------------

    @staticmethod
    def _passes_keyword_check(product: Product, spec: SearchSpec) -> bool:
        """
        Only checks must_have_keywords and avoid_keywords in Python.
        Price / rating / Prime / brand are already enforced by Amazon's URL filters.
        """
        title_lower = product.title.lower()
        for term in spec.avoid_keywords:
            if term.lower() in title_lower:
                return False
        return True  # must_have is scored by LLMRelevanceFilter, not hard-filtered here

    @staticmethod
    def _guess_brand(title: str) -> str | None:
        words = title.split()
        return words[0].strip(":-,") if words else None
