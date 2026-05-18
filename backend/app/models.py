from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class ProductFilters(BaseModel):
    query: str = ""
    min_price: float | None = None
    max_price: float | None = None
    min_rating: float | None = Field(default=None, ge=0, le=5)
    min_reviews: int | None = Field(default=None, ge=0)
    prime_only: bool = False
    brands: list[str] = Field(default_factory=list)
    must_have: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    sort_goal: Literal["best_match", "value", "rating", "budget"] = "best_match"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class Product(BaseModel):
    product_id: str
    title: str
    url: str
    price: float | None = None
    rating: float | None = None
    review_count: int | None = None
    image_url: str | None = None
    brand: str | None = None
    is_prime: bool = False
    source: str = "amazon"
    scraped_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    raw: dict[str, Any] = Field(default_factory=dict)


class RankedProduct(Product):
    score: float = 0
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    summary: str
    top_product: RankedProduct | None = None
    ranked_products: list[RankedProduct] = Field(default_factory=list)
    explanation: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    filters: ProductFilters = Field(default_factory=ProductFilters)


class ChatResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage]
    filters: ProductFilters
    recommendation: Recommendation
    products_saved: int
    trace: list[str] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    session_id: str
    user_message: str
    messages: list[dict[str, str]]
    filters: dict[str, Any]
    search_query: str
    products: list[dict[str, Any]]
    ranked_products: list[dict[str, Any]]
    recommendation: dict[str, Any]
    trace: list[str]
    products_saved: int

