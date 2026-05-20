from __future__ import annotations

"""
Amazon Catalog Scraper — v2
===========================
Key changes from v1:
  - Accepts a ``SearchSpec`` instead of ``ProductFilters``.
  - Builds the Amazon SERP URL with native filter parameters (price, rating,
    Prime, brand, sort) so Amazon applies them server-side BEFORE delivering
    results.  No more ``_passes_filters()`` post-processing.
  - Keyword-only check (must_have / avoid) is the ONLY Python-side filter
    kept, because Amazon can't filter on arbitrary keywords.
  - Query-result caching: identical SearchSpec hits are served from an
    in-process TTL cache to reduce Playwright round-trips during a session.
"""

import asyncio
import hashlib
import logging
import re
import time
from urllib.parse import quote_plus, urlencode, urljoin

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
    """Playwright-powered Amazon SERP scraper using native URL-level filters."""

    async def search(self, spec: SearchSpec) -> list[Product]:
        """
        Scrape Amazon using the SearchSpec.
        1. Build pre-filtered URL (Amazon applies server-side filters).
        2. Extract product cards with Playwright.
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

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.warning("Playwright not installed — returning sample products.")
            return sample_products(spec.search_term)

        products: list[Product] = []
        extracted_count = 0

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    viewport={"width": 1366, "height": 900},
                    user_agent=(
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                    ),
                    locale="en-US",
                )
                page = await context.new_page()
                page.set_default_timeout(settings.scraper_timeout_ms)

                max_pages = min(spec.max_pages, settings.scraper_max_pages)
                for page_number in range(1, max_pages + 1):
                    url = build_search_url(domain, spec, page_number)
                    logger.info("AmazonCatalogScraper: fetching %s", url)

                    await page.goto(url, wait_until="domcontentloaded")
                    await page.wait_for_timeout(1200)

                    cards = page.locator('[data-component-type="s-search-result"]')
                    count = await cards.count()

                    for index in range(count):
                        if len(products) >= settings.scraper_max_results:
                            break
                        product = await self._extract_card(
                            cards.nth(index), domain, currency_code, currency_symbol
                        )
                        if product:
                            extracted_count += 1
                            if self._passes_keyword_check(product, spec):
                                products.append(product)

                    if len(products) >= settings.scraper_max_results:
                        break
                    await asyncio.sleep(0.8)

                await browser.close()
        except Exception as exc:
            logger.error("AmazonCatalogScraper error: %s", exc)
            return sample_products(spec.search_term)

        if not extracted_count:
            return sample_products(spec.search_term)

        # ---- cache store ----
        _RESULT_CACHE[key] = (time.monotonic(), products)
        return products

    # ------------------------------------------------------------------
    # Card extraction
    # ------------------------------------------------------------------

    async def _extract_card(
        self,
        card,
        domain: str,
        currency_code: str,
        currency_symbol: str,
    ) -> Product | None:
        title = await self._first_text(
            card,
            [
                "h2 span",
                '[data-cy="title-recipe-title"]',
                ".a-size-medium.a-color-base.a-text-normal",
            ],
        )
        if not title:
            return None

        href = await self._first_attr(card, ["h2 a", "a.a-link-normal.s-no-outline"], "href")
        url = urljoin(domain, href) if href else domain
        product_id = await card.get_attribute("data-asin")
        if not product_id:
            product_id = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]

        price_text = await self._first_text(
            card,
            [
                ".a-price .a-offscreen",
                "[data-cy='price-recipe'] .a-price .a-offscreen",
                ".a-price",
            ],
        )
        if not price_text:
            whole = await self._first_text(card, [".a-price-whole"])
            fraction = await self._first_text(card, [".a-price-fraction"])
            if whole:
                price_text = f"{whole}.{fraction or '00'}"

        rating_text = await self._first_attr(
            card, [".a-icon-star-small", ".a-icon-star"], "aria-label"
        )
        if not rating_text:
            rating_text = await self._first_text(card, [".a-icon-alt"])

        reviews_text = await self._first_text(
            card,
            [
                "span.a-size-base.s-underline-text",
                "a[href*='customerReviews'] span",
                "[aria-label*='ratings']",
            ],
        )
        if not reviews_text:
            reviews_text = await self._first_attr(
                card, ["[aria-label*='ratings']"], "aria-label"
            )

        image_url = await self._first_attr(card, ["img.s-image"], "src")
        prime_text = await self._first_text(card, [".a-icon-prime"])
        if not prime_text:
            prime_text = await self._first_attr(
                card, ["[aria-label*='Prime']"], "aria-label"
            )

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

    # ------------------------------------------------------------------
    # DOM helpers
    # ------------------------------------------------------------------

    async def _first_text(self, card, selectors: list[str]) -> str | None:
        for selector in selectors:
            locator = card.locator(selector).first
            try:
                if await locator.count():
                    text = await locator.inner_text()
                    if text.strip():
                        return text.strip()
            except Exception:
                continue
        return None

    async def _first_attr(self, card, selectors: list[str], attr: str) -> str | None:
        for selector in selectors:
            locator = card.locator(selector).first
            try:
                if await locator.count():
                    value = await locator.get_attribute(attr)
                    if value:
                        return value
            except Exception:
                continue
        return None

    @staticmethod
    def _guess_brand(title: str) -> str | None:
        words = title.split()
        return words[0].strip(":-,") if words else None
