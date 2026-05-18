from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from .config import get_settings
from .marketplaces import marketplace_currency, marketplace_domain
from .models import Product, ProductFilters, RankedProduct


class ProductRepository(ABC):
    @abstractmethod
    async def save_products(
        self, session_id: str, products: Iterable[RankedProduct]
    ) -> int:
        raise NotImplementedError


class LocalProductRepository(ProductRepository):
    def __init__(self, path: Path | None = None) -> None:
        settings = get_settings()
        settings.local_data_dir.mkdir(parents=True, exist_ok=True)
        self.path = path or settings.local_data_dir / "products.json"

    async def save_products(
        self, session_id: str, products: Iterable[RankedProduct]
    ) -> int:
        existing = self._read()
        saved = 0
        for product in products:
            key = f"{session_id}:{product.product_id}"
            existing[key] = {
                "session_id": session_id,
                **product.model_dump(),
            }
            saved += 1
        self.path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        return saved

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}


class DynamoDBProductRepository(ProductRepository):
    def __init__(self) -> None:
        import boto3

        settings = get_settings()
        self.table = boto3.resource("dynamodb", region_name=settings.aws_region).Table(
            settings.dynamodb_table
        )

    async def save_products(
        self, session_id: str, products: Iterable[RankedProduct]
    ) -> int:
        saved = 0
        with self.table.batch_writer() as batch:
            for product in products:
                item = {
                    "pk": f"SESSION#{session_id}",
                    "sk": f"PRODUCT#{product.product_id}",
                    "session_id": session_id,
                    **json.loads(product.model_dump_json()),
                }
                batch.put_item(Item=item)
                saved += 1
        return saved


def build_product_repository() -> ProductRepository:
    settings = get_settings()
    if settings.use_dynamodb:
        try:
            return DynamoDBProductRepository()
        except Exception:
            return LocalProductRepository()
    return LocalProductRepository()


def sample_products(query: str, filters: ProductFilters | None = None) -> list[Product]:
    normalized = query or "wireless headphones"
    domain = marketplace_domain(filters.marketplace if filters else None)
    currency_code, currency_symbol = marketplace_currency(filters.marketplace if filters else None)
    preferred_brands = filters.brands if filters else []
    brands = [
        preferred_brands[0] if len(preferred_brands) > 0 else "SampleBrand",
        preferred_brands[1] if len(preferred_brands) > 1 else "ValueLine",
        preferred_brands[2] if len(preferred_brands) > 2 else "PrimeAudio",
    ]
    ratings = _sample_ratings(filters)
    reviews = _sample_reviews(filters)
    prices = _sample_prices(filters)
    prime_values = [True, True, bool(filters.prime_only) if filters else False]

    return [
        Product(
            product_id="sample-1",
            title=f"{brands[0]} {normalized.title()} - Balanced Choice",
            url=f"{domain}/s?k=" + normalized.replace(" ", "+"),
            price=prices[0],
            currency_code=currency_code,
            currency_symbol=currency_symbol,
            rating=ratings[0],
            review_count=reviews[0],
            image_url=None,
            brand=brands[0],
            is_prime=prime_values[0],
            raw={"fallback": True, "marketplace_domain": domain},
        ),
        Product(
            product_id="sample-2",
            title=f"{brands[1]} {normalized.title()} - Budget Pick",
            url=f"{domain}/s?k=" + normalized.replace(" ", "+"),
            price=prices[1],
            currency_code=currency_code,
            currency_symbol=currency_symbol,
            rating=ratings[1],
            review_count=reviews[1],
            image_url=None,
            brand=brands[1],
            is_prime=prime_values[1],
            raw={"fallback": True, "marketplace_domain": domain},
        ),
        Product(
            product_id="sample-3",
            title=f"{brands[2]} {normalized.title()} - Premium Option",
            url=f"{domain}/s?k=" + normalized.replace(" ", "+"),
            price=prices[2],
            currency_code=currency_code,
            currency_symbol=currency_symbol,
            rating=ratings[2],
            review_count=reviews[2],
            image_url=None,
            brand=brands[2],
            is_prime=prime_values[2],
            raw={"fallback": True, "marketplace_domain": domain},
        ),
    ]


def _sample_ratings(filters: ProductFilters | None) -> list[float]:
    baseline = [4.5, 4.2, 4.7]
    if filters and filters.min_rating is not None:
        return [round(max(value, filters.min_rating), 1) for value in baseline]
    return baseline


def _sample_reviews(filters: ProductFilters | None) -> list[int]:
    baseline = [8421, 3120, 12650]
    if filters and filters.min_reviews is not None:
        return [max(value, filters.min_reviews) for value in baseline]
    return baseline


def _sample_prices(filters: ProductFilters | None) -> list[float]:
    baseline = [79.99, 39.99, 149.99]
    if not filters:
        return baseline

    prices: list[float] = []
    for index, price in enumerate(baseline):
        adjusted = price
        if filters.max_price is not None and adjusted > filters.max_price:
            adjusted = max(1.0, filters.max_price - (2 * index))
        if filters.min_price is not None and adjusted < filters.min_price:
            adjusted = filters.min_price + (8 * index)
        prices.append(round(adjusted, 2))
    return prices
