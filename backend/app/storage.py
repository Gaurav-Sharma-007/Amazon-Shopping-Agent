from __future__ import annotations

from decimal import Decimal

"""
Storage Layer
=============
Provides two concerns:

1. ``ProductRepository`` — persists ranked products per session (unchanged interface).
2. ``QueryCache`` — DynamoDB-backed cache for scraped results, keyed on SearchSpec hash.
   Falls back to a local JSON file when DynamoDB is unavailable.
"""


import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from .config import get_settings
from .marketplaces import marketplace_currency, marketplace_domain
from .models import Product, ProductFilters, RankedProduct, SearchSpec


# ===========================================================================
# Product Repository (persists ranked results per session)
# ===========================================================================

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
        # Use individual put_item() instead of batch_writer() to avoid
        # 'Provided list of item keys contains duplicates' when DynamoDB's
        # internal retry mechanism resubmits UnprocessedItems alongside new ones.
        # put_item is idempotent and has no batch-level key uniqueness restriction.
        seen: set[str] = set()
        saved = 0
        for product in products:
            if product.product_id in seen:
                continue  # skip genuine duplicates in the ranked list
            seen.add(product.product_id)
            item = {
                "pk": f"SESSION#{session_id}",
                "sk": f"PRODUCT#{product.product_id}",
                "session_id": session_id,
                # parse_float=Decimal: DynamoDB rejects Python float — must be Decimal.
                **json.loads(product.model_dump_json(), parse_float=Decimal),
            }
            self.table.put_item(Item=item)
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


# ===========================================================================
# Query Cache  (caches raw scrape results to avoid redundant browser calls)
# ===========================================================================

class QueryCache(ABC):
    """Abstract TTL cache for raw Product lists keyed on SearchSpec."""

    @abstractmethod
    def get(self, cache_key: str) -> list[Product] | None:
        raise NotImplementedError

    @abstractmethod
    def put(self, cache_key: str, products: list[Product], ttl_seconds: int) -> None:
        raise NotImplementedError


class LocalQueryCache(QueryCache):
    """In-process dict cache with TTL.  Survives only for the process lifetime."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, list[Product]]] = {}

    def get(self, cache_key: str) -> list[Product] | None:
        entry = self._store.get(cache_key)
        if entry is None:
            return None
        expires_at, products = entry
        if time.monotonic() > expires_at:
            del self._store[cache_key]
            return None
        return products

    def put(self, cache_key: str, products: list[Product], ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        self._store[cache_key] = (time.monotonic() + ttl_seconds, products)


class DynamoDBQueryCache(QueryCache):
    """
    DynamoDB-backed cache.  Each item has a DynamoDB TTL attribute so AWS
    automatically purges expired entries.

    Table schema (must exist or be created separately):
      pk  (S)  — "QUERYCACHE#<cache_key>"
      sk  (S)  — "RESULT"
      ttl (N)  — Unix epoch expiry (DynamoDB TTL attribute name must be 'ttl')
      data (S) — JSON-encoded list[Product]
    """

    def __init__(self) -> None:
        import boto3

        settings = get_settings()
        self._table = boto3.resource("dynamodb", region_name=settings.aws_region).Table(
            settings.dynamodb_table
        )

    def get(self, cache_key: str) -> list[Product] | None:
        try:
            response = self._table.get_item(
                Key={"pk": f"QUERYCACHE#{cache_key}", "sk": "RESULT"}
            )
            item = response.get("Item")
            if not item:
                return None
            # DynamoDB TTL expiry is eventually consistent; do a local time check too
            if int(item.get("ttl", 0)) < int(time.time()):
                return None
            return [Product(**p) for p in json.loads(item["data"])]
        except Exception:
            return None

    def put(self, cache_key: str, products: list[Product], ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        try:
            self._table.put_item(
                Item={
                    "pk": f"QUERYCACHE#{cache_key}",
                    "sk": "RESULT",
                    "ttl": int(time.time()) + ttl_seconds,
                    "data": json.dumps([p.model_dump() for p in products]),
                }
            )
        except Exception:
            pass  # cache is best-effort; never fail the main pipeline


def build_query_cache() -> QueryCache:
    """Return DynamoDB cache if available, else in-process fallback."""
    settings = get_settings()
    if settings.use_dynamodb:
        try:
            return DynamoDBQueryCache()
        except Exception:
            pass
    return LocalQueryCache()


# ===========================================================================
# Sample products (used when scraper is unavailable — dev/test fallback)
# ===========================================================================

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
        return [round(max(v, filters.min_rating), 1) for v in baseline]
    return baseline


def _sample_reviews(filters: ProductFilters | None) -> list[int]:
    baseline = [8421, 3120, 12650]
    if filters and filters.min_reviews is not None:
        return [max(v, filters.min_reviews) for v in baseline]
    return baseline


def _sample_prices(filters: ProductFilters | None) -> list[float]:
    baseline = [79.99, 39.99, 149.99]
    if not filters:
        return baseline
    prices: list[float] = []
    for i, price in enumerate(baseline):
        adj = price
        if filters.max_price is not None and adj > filters.max_price:
            adj = max(1.0, filters.max_price - (2 * i))
        if filters.min_price is not None and adj < filters.min_price:
            adj = filters.min_price + (8 * i)
        prices.append(round(adj, 2))
    return prices
