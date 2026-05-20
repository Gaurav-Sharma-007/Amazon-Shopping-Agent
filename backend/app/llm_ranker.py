from __future__ import annotations

"""
LLM Ranker
==========
Ranks the relevance-filtered products using Amazon Bedrock.
Claude scores each product across three axes:
  - value_for_money  (price vs. specs)
  - user_intent_fit  (how well it matches the SearchSpec)
  - quality_signals  (rating, review count)

Returns a list of ``RankedProduct`` objects sorted best-first, each with an
``llm_rank_reasoning`` string explaining the rank.

Falls back to the deterministic ``ranking.py`` scorer if Bedrock is unavailable.
"""

import json
import logging
import math
import re

from .config import get_settings
from .models import ProductFilters, RankedProduct, ScoredProduct, SearchSpec

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert product ranker for an Amazon recommender system.
You will receive a user's SearchSpec and a list of pre-filtered product candidates.

Rank the products from best to worst and return ONLY a JSON array in this format:
[
  {
    "product_id": "<id>",
    "rank": 1,
    "scores": {
      "value_for_money": <0-10>,
      "user_intent_fit": <0-10>,
      "quality_signals": <0-10>
    },
    "reasoning": "<2-3 sentence explanation for this rank>"
  },
  ...
]

Scoring guide:
- value_for_money: compare price vs. specs; penalise overpriced, reward budget options.
- user_intent_fit: how well does this product match the search_term and must_have_keywords?
- quality_signals: based on rating (out of 5) and review count.

Sort the array by rank ascending (rank 1 = best).
Return ONLY the JSON array, no other text.
"""


class LLMRanker:
    """
    LLM-driven ranker.  Primary: Bedrock Claude.  Fallback: deterministic scorer.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    async def rank(
        self,
        scored_products: list[ScoredProduct],
        spec: SearchSpec,
        filters: ProductFilters,
    ) -> list[RankedProduct]:
        """Return RankedProduct list, best first."""
        if not scored_products:
            return []

        llm_result = await self._rank_with_bedrock(scored_products, spec)
        if llm_result is not None:
            logger.info("LLMRanker: Bedrock ranking succeeded.")
            return llm_result

        logger.warning("LLMRanker: Bedrock unavailable, using deterministic fallback.")
        return self._deterministic_fallback(scored_products, filters)

    # ------------------------------------------------------------------
    # Bedrock path
    # ------------------------------------------------------------------

    async def _rank_with_bedrock(
        self, products: list[ScoredProduct], spec: SearchSpec
    ) -> list[RankedProduct] | None:
        try:
            import boto3  # noqa: PLC0415
        except ImportError:
            return None

        payload = {
            "search_spec": {
                "search_term": spec.search_term,
                "must_have_keywords": spec.must_have_keywords,
                "sort_goal": spec.sort_goal,
                "brands": spec.brands,
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
                    "relevance_score": p.relevance_score,
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
            raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.I)
            raw = re.sub(r"\n?```$", "", raw)
            ranks: list[dict] = json.loads(raw)

            # Build map: product_id → rank entry
            rank_map = {entry["product_id"]: entry for entry in ranks}
            product_map = {p.product_id: p for p in products}

            result: list[RankedProduct] = []
            for entry in sorted(ranks, key=lambda e: e.get("rank", 99)):
                pid = entry["product_id"]
                product = product_map.get(pid)
                if not product:
                    continue
                scores = entry.get("scores", {})
                total = sum(scores.values())
                result.append(
                    RankedProduct(
                        **{k: v for k, v in product.model_dump().items()
                           if k not in {"relevance_score", "relevance_reasoning", "is_relevant"}},
                        score=round(total, 2),
                        score_breakdown={k: round(float(v), 2) for k, v in scores.items()},
                        reasons=[entry.get("reasoning", "")],
                        llm_rank_reasoning=entry.get("reasoning", ""),
                    )
                )
            return result
        except Exception as exc:
            logger.warning("LLMRanker Bedrock error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Deterministic fallback (adapted from ranking.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _deterministic_fallback(
        products: list[ScoredProduct], filters: ProductFilters
    ) -> list[RankedProduct]:
        prices = [p.price for p in products if p.price is not None]
        min_price = min(prices) if prices else 0
        spread = max((max(prices) if prices else 1) - min_price, 1)

        result: list[RankedProduct] = []
        for product in products:
            rating_s = ((product.rating or 3.5) / 5) * 10
            review_s = min(math.log10((product.review_count or 1) + 1) / 5, 1) * 10
            if product.price is not None:
                price_s = (1 - (product.price - min_price) / spread) * 10
            else:
                price_s = 5.0
            relevance_s = product.relevance_score  # already 0-10
            prime_s = 2.0 if product.is_prime else 0.0

            breakdown = {
                "quality_signals": round((rating_s + review_s) / 2, 2),
                "value_for_money": round(price_s, 2),
                "user_intent_fit": round(relevance_s, 2),
                "prime_bonus": prime_s,
            }
            total = round(sum(breakdown.values()), 2)
            result.append(
                RankedProduct(
                    **{k: v for k, v in product.model_dump().items()
                       if k not in {"relevance_score", "relevance_reasoning", "is_relevant"}},
                    score=total,
                    score_breakdown=breakdown,
                    reasons=[f"Relevance: {product.relevance_score:.1f}/10  "
                              f"Rating: {product.rating or 'N/A'}  "
                              f"Reviews: {product.review_count or 0:,}"],
                    llm_rank_reasoning="deterministic-fallback",
                )
            )
        return sorted(result, key=lambda p: p.score, reverse=True)
