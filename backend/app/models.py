from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator

from .marketplaces import DEFAULT_MARKETPLACE_CODE, normalize_marketplace_code


# ---------------------------------------------------------------------------
# Filter & Search Spec models
# ---------------------------------------------------------------------------

class AmazonURLFilters(BaseModel):
    """Native Amazon SERP URL query parameters applied before page load."""

    p_36: str | None = None   # price range  e.g. "100-200000" (units × 100)
    p_72: str | None = None   # min-rating node ID  e.g. "2421889011" = 4★+
    p_85: str | None = None   # prime eligible  "2470955011"
    p_89: str | None = None   # brand filter  e.g. "Sony"
    s: str | None = None      # sort  review-rank | price-asc-rank | price-desc-rank
    i: str | None = None      # department slug  e.g. "electronics"

    def to_query_params(self) -> dict[str, str]:
        """Return only the non-None params as a plain dict for urlencode."""
        return {k.replace("_", "-") if k.startswith("p_") else k: v
                for k, v in self.model_dump().items() if v is not None}


class SearchSpec(BaseModel):
    """
    Structured scrape plan produced by the LLM Query Planner.
    Drives both the URL builder and the post-fetch keyword checks.
    """

    search_term: str
    category_node: str | None = None
    marketplace: str = DEFAULT_MARKETPLACE_CODE
    amazon_url_filters: AmazonURLFilters = Field(default_factory=AmazonURLFilters)
    must_have_keywords: list[str] = Field(default_factory=list)
    avoid_keywords: list[str] = Field(default_factory=list)
    brands: list[str] = Field(default_factory=list)
    sort_goal: Literal["best_match", "value", "rating", "budget"] = "best_match"
    max_pages: int = Field(default=1, ge=1, le=5)

    @field_validator("marketplace", mode="before")
    @classmethod
    def normalize_marketplace(cls, value: str | None) -> str:
        return normalize_marketplace_code(value)


class ProductFilters(BaseModel):
    """
    Kept for backward compatibility with session memory and the API surface.
    Internally converted to SearchSpec before scraping.
    """

    query: str = ""
    marketplace: str = DEFAULT_MARKETPLACE_CODE
    min_price: float | None = None
    max_price: float | None = None
    min_rating: float | None = Field(default=None, ge=0, le=5)
    min_reviews: int | None = Field(default=None, ge=0)
    prime_only: bool = False
    brands: list[str] = Field(default_factory=list)
    must_have: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    sort_goal: Literal["best_match", "value", "rating", "budget"] = "best_match"

    @field_validator("marketplace", mode="before")
    @classmethod
    def normalize_marketplace(cls, value: str | None) -> str:
        return normalize_marketplace_code(value)


# ---------------------------------------------------------------------------
# Product models
# ---------------------------------------------------------------------------

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
    currency_code: str = "USD"
    currency_symbol: str = "$"
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


class ScoredProduct(Product):
    """Product annotated with an LLM relevance score (0–10) and drop flag."""

    relevance_score: float = 0.0
    relevance_reasoning: str = ""
    is_relevant: bool = True


class RankedProduct(Product):
    score: float = 0
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    llm_rank_reasoning: str = ""


# ---------------------------------------------------------------------------
# Chat / API models
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# LangGraph agent state
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    session_id: str
    user_message: str
    messages: list[dict[str, str]]
    filters: dict[str, Any]           # ProductFilters dict (session compat)
    scrape_plan: dict[str, Any]       # SearchSpec dict (drives scraper)
    search_query: str
    products: list[dict[str, Any]]
    scored_products: list[dict[str, Any]]   # after LLM relevance filter
    ranked_products: list[dict[str, Any]]
    recommendation: dict[str, Any]
    trace: list[str]
    products_saved: int
    cache_hit: bool
