from __future__ import annotations

import asyncio
import hashlib
import re
from urllib.parse import quote_plus, urljoin

from .config import get_settings
from .models import Product, ProductFilters
from .storage import sample_products


PRICE_RE = re.compile(r"(\d+(?:,\d{3})*(?:\.\d{2})?)")
RATING_RE = re.compile(r"([0-5](?:\.\d)?)\s+out of 5")
REVIEWS_RE = re.compile(r"(\d[\d,]*)")


def _parse_price(value: str | None) -> float | None:
    if not value:
        return None
    match = PRICE_RE.search(value.replace("\n", "."))
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


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


class AmazonCatalogScraper:
    """Playwright traversal for public Amazon search result pages."""

    async def search(self, filters: ProductFilters) -> list[Product]:
        settings = get_settings()
        query = filters.query.strip() or "wireless headphones"
        products: list[Product] = []

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return sample_products(query)

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(viewport={"width": 1366, "height": 900})
                page.set_default_timeout(settings.scraper_timeout_ms)

                for page_number in range(1, settings.scraper_max_pages + 1):
                    url = (
                        f"{settings.amazon_domain}/s?k={quote_plus(query)}"
                        f"&page={page_number}"
                    )
                    await page.goto(url, wait_until="domcontentloaded")
                    await page.wait_for_timeout(1200)

                    cards = page.locator('[data-component-type="s-search-result"]')
                    count = await cards.count()
                    for index in range(count):
                        if len(products) >= settings.scraper_max_results:
                            break
                        product = await self._extract_card(cards.nth(index), settings.amazon_domain)
                        if product and self._passes_filters(product, filters):
                            products.append(product)

                    if len(products) >= settings.scraper_max_results:
                        break
                    await asyncio.sleep(0.8)

                await browser.close()
        except Exception:
            return sample_products(query)

        return products or sample_products(query)

    async def _extract_card(self, card, domain: str) -> Product | None:
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
            reviews_text = await self._first_attr(card, ["[aria-label*='ratings']"], "aria-label")
        image_url = await self._first_attr(card, ["img.s-image"], "src")
        prime_text = await self._first_text(card, [".a-icon-prime"])
        if not prime_text:
            prime_text = await self._first_attr(card, ["[aria-label*='Prime']"], "aria-label")

        return Product(
            product_id=product_id,
            title=" ".join(title.split()),
            url=url,
            price=_parse_price(price_text),
            rating=_parse_rating(rating_text),
            review_count=_parse_reviews(reviews_text),
            image_url=image_url,
            brand=self._guess_brand(title),
            is_prime=bool(prime_text),
            raw={"price_text": price_text, "rating_text": rating_text},
        )

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

    def _passes_filters(self, product: Product, filters: ProductFilters) -> bool:
        if filters.min_price is not None and product.price is not None:
            if product.price < filters.min_price:
                return False
        if filters.max_price is not None and product.price is not None:
            if product.price > filters.max_price:
                return False
        if filters.min_rating is not None and product.rating is not None:
            if product.rating < filters.min_rating:
                return False
        if filters.min_reviews is not None and product.review_count is not None:
            if product.review_count < filters.min_reviews:
                return False
        if filters.prime_only and not product.is_prime:
            return False
        if filters.brands and product.brand:
            brands = {brand.lower() for brand in filters.brands}
            if product.brand.lower() not in brands:
                return False
        avoid = [item.lower() for item in filters.avoid]
        if any(term in product.title.lower() for term in avoid):
            return False
        return True

    def _guess_brand(self, title: str) -> str | None:
        words = title.split()
        if not words:
            return None
        return words[0].strip(":-,")
