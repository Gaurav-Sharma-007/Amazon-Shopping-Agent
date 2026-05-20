from __future__ import annotations

"""
LLM Query Planner
=================
Converts a raw user message into a structured ``SearchSpec`` using Amazon
Bedrock (Claude).  Falls back gracefully to a regex-based extraction so the
pipeline never hard-fails when Bedrock is unavailable.

SearchSpec fields map directly to Amazon URL query parameters so that
ALL major filters (price, rating, Prime, brand, sort) are applied on
Amazon's search results page *before* Playwright loads any cards.
"""

import json
import logging
import re
from urllib.parse import quote_plus

from .config import get_settings
from .marketplaces import (
    amazon_rating_node_id,
    find_marketplace_in_text,
    marketplace_currency,
    normalize_marketplace_code,
    strip_marketplace_terms,
)
from .models import AmazonURLFilters, ProductFilters, SearchSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex fallback patterns (same as old intent_agent — kept as safety net)
# ---------------------------------------------------------------------------
_PRICE_RANGE_RE = re.compile(
    r"(?:price\s*)?(?:range\s*)?(?:of|between|from)?\s*(?:[$₹]|rs\.?|inr)?\s*"
    r"(\d[\d,]*(?:\.\d+)?)\s*(?:to|-|and)\s*(?:[$₹]|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)",
    re.I,
)
_PRICE_UNDER_RE = re.compile(
    r"(?:under|below|less than|max|budget)\s*(?:[$₹]|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)(?![\d.])(?!\s*(?:stars?|reviews?|ratings?))",
    re.I,
)
_PRICE_OVER_RE = re.compile(
    r"(?:over|above|at least|min(?:imum)?)\s*(?:[$₹]|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)(?![\d.])(?!\s*(?:stars?|reviews?|ratings?))",
    re.I,
)
_RATING_RE = re.compile(r"(\d(?:\.\d)?)\s*(?:star|stars|\+)", re.I)
_RATING_KW_RE = re.compile(
    r"(?:min(?:imum)?\s*)?(?:rating|rated)\s*(?:of|at least|above|over)?\s*(\d(?:\.\d)?)",
    re.I,
)
_BRAND_RE = re.compile(r"\bbrands?\s*(?:are|include|:|=)?\s*([a-z0-9][a-z0-9&\-\s,]+)", re.I)
_FROM_RE = re.compile(r"\bfrom\s+([a-z0-9][a-z0-9&\-\s,]+)", re.I)

_SORT_GOAL_MAP: dict[str, str] = {
    "review-rank": "rating",
    "price-asc-rank": "budget",
    "price-desc-rank": "best_match",
    "exact-aware-popularity-rank": "best_match",
}

# ---------------------------------------------------------------------------
# System prompt fed to Claude
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are an Amazon product search specialist.
Given a user's natural language request, return ONLY a valid JSON object
matching the schema below — no explanation, no markdown fences.

Schema:
{
  "search_term": "concise Amazon search noun phrase (no price / rating words)",
  "category_node": "electronics | clothing | home | sports | books | etc. or null",
  "amazon_url_filters": {
    "p_36": "<min_currency_units_x100>-<max_currency_units_x100> or null",
    "p_72": "<rating_node_id> or null",
    "p_85": "2470955011 if Prime requested, else null",
    "p_89": "<Single brand name> or null",
    "s":    "review-rank | price-asc-rank | price-desc-rank | null",
    "i":    "<amazon dept slug e.g. electronics> or null"
  },
  "must_have_keywords": ["feature1"],
  "avoid_keywords": ["term1"],
  "brands": ["Brand1", "Brand2"],
  "sort_goal": "best_match | value | rating | budget",
  "max_pages": 1
}

Rules:
- search_term: clean product noun phrase only — strip price, rating, Prime words.
- p_36 values are in smallest currency unit × 100.
  Examples: ₹1500 → 150000,  $20 → 2000. Use "1" as min if only max is given.
- p_72 rating node IDs (use these EXACTLY):
    amazon.in / amazon.com  : 4★+ = 2421889011, 3★+ = 2421890011
    amazon.co.uk            : 4★+ = 328520031,  3★+ = 328521031
    amazon.de               : 4★+ = 669342031,  3★+ = 669343031
- p_89: set only when a SINGLE brand is requested; leave null for multiple.
- s: choose review-rank when quality matters, price-asc-rank for budget queries.
- max_pages: 1 unless user asks for many options (max 3).
"""


# ---------------------------------------------------------------------------
# Main planner class
# ---------------------------------------------------------------------------

class LLMQueryPlanner:
    """
    Generates a ``SearchSpec`` from a user message.
    Primary path: Bedrock Claude (structured JSON).
    Fallback path: regex extraction (mirrors old intent_agent behaviour).
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    async def plan(
        self,
        user_message: str,
        session_messages: list[dict],
        current_filters: ProductFilters,
    ) -> SearchSpec:
        """Return a SearchSpec.  Never raises — falls back to regex on error."""
        bedrock_result = await self._call_bedrock(user_message, session_messages)
        if bedrock_result:
            logger.info("LLMQueryPlanner: Bedrock plan succeeded.")
            # Honour marketplace from message even if LLM missed it
            marketplace = find_marketplace_in_text(user_message) or current_filters.marketplace
            bedrock_result.marketplace = normalize_marketplace_code(marketplace)
            return bedrock_result

        logger.warning("LLMQueryPlanner: Bedrock unavailable, using regex fallback.")
        return self._regex_fallback(user_message, current_filters)

    # ------------------------------------------------------------------
    # Bedrock path
    # ------------------------------------------------------------------

    async def _call_bedrock(
        self, user_message: str, session_messages: list[dict]
    ) -> SearchSpec | None:
        try:
            import boto3  # noqa: PLC0415
        except ImportError:
            return None

        conversation: list[dict] = []
        for msg in session_messages[-6:]:  # last 3 turns for context
            role = msg.get("role", "user")
            if role in {"user", "assistant"}:
                conversation.append({
                    "role": role,
                    "content": [{"text": msg.get("content", "")}],
                })
        conversation.append({
            "role": "user",
            "content": [{"text": user_message}],
        })

        try:
            client = boto3.client("bedrock-runtime", region_name=self._settings.aws_region)
            response = client.converse(
                modelId=self._settings.bedrock_model_id,
                system=[{"text": _SYSTEM_PROMPT}],
                messages=conversation,
            )
            raw_text: str = response["output"]["message"]["content"][0]["text"]
            # strip possible markdown fences
            raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text.strip(), flags=re.I)
            raw_text = re.sub(r"\n?```$", "", raw_text.strip())
            data = json.loads(raw_text)
            url_filters = AmazonURLFilters(**data.pop("amazon_url_filters", {}))
            return SearchSpec(amazon_url_filters=url_filters, **data)
        except Exception as exc:
            logger.warning("LLMQueryPlanner Bedrock error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Regex fallback (mirrors old _intent_agent logic)
    # ------------------------------------------------------------------

    def _regex_fallback(self, message: str, current: ProductFilters) -> SearchSpec:
        lower = message.lower()
        url_filters = AmazonURLFilters()
        sort_goal: str = current.sort_goal

        # ---- price ----
        range_m = _PRICE_RANGE_RE.search(message)
        under_m = _PRICE_UNDER_RE.search(message)
        over_m = _PRICE_OVER_RE.search(message)

        min_price: float | None = current.min_price
        max_price: float | None = current.max_price

        if range_m:
            lo = float(range_m.group(1).replace(",", ""))
            hi = float(range_m.group(2).replace(",", ""))
            min_price, max_price = min(lo, hi), max(lo, hi)
        else:
            if under_m:
                max_price = float(under_m.group(1).replace(",", ""))
            if over_m:
                min_price = float(over_m.group(1).replace(",", ""))

        if min_price is not None or max_price is not None:
            lo_units = int((min_price or 1) * 100)
            hi_units = int((max_price or 9_999_999) * 100)
            url_filters.p_36 = f"{lo_units}-{hi_units}"

        # ---- rating ----
        rating_m = _RATING_RE.search(message) or _RATING_KW_RE.search(message)
        min_rating: float | None = current.min_rating
        if rating_m:
            min_rating = min(5.0, float(rating_m.group(1)))
        if min_rating is not None:
            url_filters.p_72 = amazon_rating_node_id(current.marketplace, min_rating)

        # ---- prime ----
        if "prime" in lower or current.prime_only:
            url_filters.p_85 = "2470955011"

        # ---- brand ----
        brands: list[str] = list(current.brands)
        brand_m = _BRAND_RE.search(message) or _FROM_RE.search(message)
        if brand_m:
            for part in re.split(r",|/|\bor\b|\band\b", brand_m.group(1), flags=re.I):
                brand = " ".join(part.strip(" .:-").split())
                if brand and len(brand.split()) <= 3:
                    brands.append(brand.title())
        if len(brands) == 1:
            url_filters.p_89 = quote_plus(brands[0])

        # ---- sort ----
        if any(w in lower for w in ["cheap", "budget", "affordable"]):
            sort_goal = "budget"
            url_filters.s = "price-asc-rank"
        elif any(w in lower for w in ["top rated", "best rated", "highest rated"]):
            sort_goal = "rating"
            url_filters.s = "review-rank"
        elif any(w in lower for w in ["value", "best for money", "worth"]):
            sort_goal = "value"

        # ---- search term ----
        search_term = self._clean_search_term(message, current.query)

        # ---- must-have / avoid keywords ----
        must_kw = list(current.must_have)
        avoid_kw = list(current.avoid)
        for kw in re.split(r",| and ", re.split(r"\b(?:with|must have)\b", lower, maxsplit=1)[-1]):
            kw = kw.strip(" .")
            if kw and len(kw.split()) <= 4 and kw not in must_kw:
                must_kw.append(kw)
        for kw in re.split(r",| and ", re.split(r"\b(?:avoid|without|exclude)\b", lower, maxsplit=1)[-1]):
            kw = kw.strip(" .")
            if kw and len(kw.split()) <= 4 and kw not in avoid_kw:
                avoid_kw.append(kw)

        marketplace = find_marketplace_in_text(message) or current.marketplace

        return SearchSpec(
            search_term=search_term or current.query or "product",
            marketplace=marketplace,
            amazon_url_filters=url_filters,
            must_have_keywords=must_kw[:5],
            avoid_keywords=avoid_kw[:5],
            brands=brands[:5],
            sort_goal=sort_goal,  # type: ignore[arg-type]
        )

    @staticmethod
    def _clean_search_term(message: str, current_query: str) -> str:
        cleaned = strip_marketplace_terms(message)
        cleaned = _PRICE_RANGE_RE.sub("", cleaned)
        cleaned = _PRICE_UNDER_RE.sub("", cleaned)
        cleaned = _PRICE_OVER_RE.sub("", cleaned)
        cleaned = _RATING_RE.sub("", cleaned)
        cleaned = _RATING_KW_RE.sub("", cleaned)
        cleaned = re.sub(
            r"\b(i want|i need|find me|recommend|show me|looking for|best|amazon|prime|"
            r"cheap|budget|affordable|with|must have|avoid|without|exclude)\b",
            "",
            cleaned,
            flags=re.I,
        )
        cleaned = re.sub(r"\b\d+\+?\s*(?:reviews?|ratings?)\b", "", cleaned, flags=re.I)
        cleaned = " ".join(cleaned.split()).strip(" ,.-")[:120]
        if not cleaned or len(cleaned) < 3:
            return current_query
        return cleaned
