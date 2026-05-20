from __future__ import annotations

"""
LLM Relevance Filter
====================
After the scraper hands back a raw ``Product`` list, this module uses
Amazon Bedrock to score each product against the user's original intent
(as captured in the ``SearchSpec``).  Products with a relevance score
below the threshold are dropped before ranking.

This replaces the old ``_passes_filters()`` / ``_satisfies_hard_filters()``
post-processing and adds semantic understanding (e.g. "ANC" matching
"Active Noise Cancellation" in a product title).
"""

import json
import logging
from typing import TYPE_CHECKING

from .config import get_settings
from .models import Product, ScoredProduct, SearchSpec

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Minimum relevance score (0–10) to keep a product.
# Kept intentionally low so border-line products still reach the ranker.
# The ranker (not this filter) is responsible for ordering by quality.
RELEVANCE_THRESHOLD = 2.5

_SYSTEM_PROMPT = """You are a product relevance judge for an Amazon recommender system.
You will receive a user's search intent (SearchSpec) and a list of product candidates.
For EACH product return a relevance score from 0 to 10 and a one-line reasoning.

Output ONLY a JSON array, one entry per product, in the SAME ORDER as input:
[
  {"product_id": "<id>", "relevance_score": <0-10>, "reasoning": "<one line>"},
  ...
]

Scoring guide — be GENEROUS; this is a pre-filter, not the final rank:
10 — Perfect match (right category, all keywords, brand)
7–9 — Very good match (right category, most features match)
5–6 — Acceptable match (correct category, some features match)
3–4 — Weak but related (category adjacent, could be useful)
1–2 — Tangentially related — only score 1-2 for clearly wrong categories
0   — Completely irrelevant (e.g. a phone case when headphones were asked for)

Important rules:
- Default to scoring 5 or above when in doubt — prefer to keep products.
- Only use must_have_keywords to penalise if they are completely absent AND critical.
- Score 0 only for avoid_keywords that actually appear in the product title.
- Do NOT penalise missing minor attributes heavily; save that for the ranker.
Return ONLY the JSON array, nothing else.
"""


class LLMRelevanceFilter:
    """
    Scores a batch of products against a SearchSpec using Bedrock.
    Falls back to a deterministic keyword check if Bedrock is unavailable.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    async def filter(
        self, products: list[Product], spec: SearchSpec
    ) -> list[ScoredProduct]:
        """Return products above the relevance threshold, ordered by score desc."""
        if not products:
            return []

        scored = await self._score_with_bedrock(products, spec)
        if scored is None:
            logger.warning("LLMRelevanceFilter: Bedrock unavailable, using keyword fallback.")
            scored = self._keyword_fallback(products, spec)

        relevant = [p for p in scored if p.is_relevant]
        # Always pass at least 5 products to the ranker so it has a meaningful
        # pool to work with, even if the LLM scored everything conservatively.
        if len(relevant) < 5 and scored:
            relevant = sorted(scored, key=lambda p: p.relevance_score, reverse=True)[:5]
            for p in relevant:
                p.is_relevant = True

        return relevant

    # ------------------------------------------------------------------
    # Bedrock path — batch all products in one call
    # ------------------------------------------------------------------

    async def _score_with_bedrock(
        self, products: list[Product], spec: SearchSpec
    ) -> list[ScoredProduct] | None:
        try:
            import boto3  # noqa: PLC0415
        except ImportError:
            return None

        payload = {
            "search_spec": {
                "search_term": spec.search_term,
                "must_have_keywords": spec.must_have_keywords,
                "avoid_keywords": spec.avoid_keywords,
                "brands": spec.brands,
                "sort_goal": spec.sort_goal,
            },
            "products": [
                {
                    "product_id": p.product_id,
                    "title": p.title,
                    "brand": p.brand,
                    "price": p.price,
                    "currency_code": p.currency_code,
                    "rating": p.rating,
                    "review_count": p.review_count,
                    "is_prime": p.is_prime,
                }
                for p in products
            ],
        }

        try:
            client = boto3.client("bedrock-runtime", region_name=self._settings.aws_region)
            response = client.converse(
                modelId=self._settings.bedrock_model_id,
                system=[{"text": _SYSTEM_PROMPT}],
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": json.dumps(payload)}],
                    }
                ],
            )
            raw = response["output"]["message"]["content"][0]["text"].strip()
            # Strip markdown fences if present
            import re
            raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.I)
            raw = re.sub(r"\n?```$", "", raw)
            scores: list[dict] = json.loads(raw)

            # Build lookup by product_id
            score_map = {entry["product_id"]: entry for entry in scores}
            result: list[ScoredProduct] = []
            for product in products:
                entry = score_map.get(product.product_id, {})
                relevance = float(entry.get("relevance_score", 5.0))
                result.append(
                    ScoredProduct(
                        **product.model_dump(),
                        relevance_score=relevance,
                        relevance_reasoning=entry.get("reasoning", ""),
                        is_relevant=relevance >= RELEVANCE_THRESHOLD,
                    )
                )
            return result
        except Exception as exc:
            logger.warning("LLMRelevanceFilter Bedrock error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Keyword fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_fallback(
        products: list[Product], spec: SearchSpec
    ) -> list[ScoredProduct]:
        result: list[ScoredProduct] = []
        for product in products:
            title_lower = product.title.lower()
            score = 5.0  # neutral baseline

            # Penalise avoid keywords hard
            for kw in spec.avoid_keywords:
                if kw.lower() in title_lower:
                    score = 0.0
                    break

            if score > 0:
                # Boost for must-have matches
                for kw in spec.must_have_keywords:
                    if kw.lower() in title_lower:
                        score = min(10.0, score + 1.5)

                # Boost for brand match
                if spec.brands and product.brand:
                    if product.brand.lower() in {b.lower() for b in spec.brands}:
                        score = min(10.0, score + 1.5)

            result.append(
                ScoredProduct(
                    **product.model_dump(),
                    relevance_score=score,
                    relevance_reasoning="keyword-fallback",
                    is_relevant=score >= RELEVANCE_THRESHOLD,
                )
            )
        return result
