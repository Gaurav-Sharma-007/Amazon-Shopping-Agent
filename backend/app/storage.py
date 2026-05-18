from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from .config import get_settings
from .models import Product, RankedProduct


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


def sample_products(query: str) -> list[Product]:
    normalized = query or "wireless headphones"
    return [
        Product(
            product_id="sample-1",
            title=f"{normalized.title()} - Balanced Choice",
            url="https://www.amazon.com/s?k=" + normalized.replace(" ", "+"),
            price=79.99,
            rating=4.5,
            review_count=8421,
            image_url=None,
            brand="SampleBrand",
            is_prime=True,
            raw={"fallback": True},
        ),
        Product(
            product_id="sample-2",
            title=f"{normalized.title()} - Budget Pick",
            url="https://www.amazon.com/s?k=" + normalized.replace(" ", "+"),
            price=39.99,
            rating=4.2,
            review_count=3120,
            image_url=None,
            brand="ValueLine",
            is_prime=True,
            raw={"fallback": True},
        ),
        Product(
            product_id="sample-3",
            title=f"{normalized.title()} - Premium Option",
            url="https://www.amazon.com/s?k=" + normalized.replace(" ", "+"),
            price=149.99,
            rating=4.7,
            review_count=12650,
            image_url=None,
            brand="PrimeAudio",
            is_prime=False,
            raw={"fallback": True},
        ),
    ]

