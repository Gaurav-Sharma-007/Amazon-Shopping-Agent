from __future__ import annotations

import json
import os

from .config import get_settings
from .models import ProductFilters, Recommendation, RankedProduct


class RecommendationLLM:
    def __init__(self) -> None:
        self.last_status = "Bedrock has not been called yet."

    async def explain(
        self, filters: ProductFilters, ranked_products: list[RankedProduct]
    ) -> Recommendation:
        top = ranked_products[0] if ranked_products else None
        if not top:
            self.last_status = "Skipped Bedrock because no ranked products were available."
            return Recommendation(
                summary="I could not find a confident product match yet.",
                explanation=["Try widening your filters or describing the product category."],
                follow_up_questions=["What budget range should I optimize for?"],
            )

        bedrock = await self._try_bedrock(filters, ranked_products[:5])
        if bedrock:
            return bedrock

        self.last_status = f"{self.last_status} Falling back to deterministic local explanation."
        return Recommendation(
            summary=f"My top pick is {top.title}.",
            top_product=top,
            ranked_products=ranked_products,
            explanation=[
                f"It scored {top.score:.1f} using rating, review volume, price fit, and your stated preferences.",
                *top.reasons[:3],
            ],
            follow_up_questions=_followups(filters),
        )

    async def _try_bedrock(
        self, filters: ProductFilters, products: list[RankedProduct]
    ) -> Recommendation | None:
        try:
            import boto3
        except ImportError:
            self.last_status = "Skipped Bedrock because boto3 is not installed."
            return None

        settings = get_settings()
        token_available = bool(os.getenv("AWS_BEARER_TOKEN_BEDROCK"))
        prompt = {
            "task": "Explain the best Amazon product recommendation with concise, transparent reasoning.",
            "filters": filters.model_dump(),
            "products": [product.model_dump() for product in products],
            "output": {
                "summary": "string",
                "explanation": ["short reason strings"],
                "follow_up_questions": ["short question strings"],
            },
        }

        try:
            client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
            response = client.converse(
                modelId=settings.bedrock_model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": (
                                    "Return recommendation prose for this JSON. "
                                    "Do not invent facts beyond the product fields.\n"
                                    + json.dumps(prompt)
                                )
                            }
                        ],
                    }
                ],
            )
            text = response["output"]["message"]["content"][0]["text"]
            top = products[0] if products else None
            self.last_status = (
                f"Used Amazon Bedrock model {settings.bedrock_model_id} "
                f"in {settings.aws_region}."
            )
            return Recommendation(
                summary=text.strip().splitlines()[0][:500],
                top_product=top,
                ranked_products=products,
                explanation=[line.strip("- ") for line in text.splitlines()[1:5] if line.strip()],
                follow_up_questions=_followups(filters),
            )
        except Exception as exc:
            token_hint = "present" if token_available else "missing"
            self.last_status = (
                f"Bedrock call failed for {settings.bedrock_model_id} "
                f"in {settings.aws_region}; AWS_BEARER_TOKEN_BEDROCK is {token_hint}; "
                f"{type(exc).__name__}: {exc}"
            )
            return None


def _followups(filters: ProductFilters) -> list[str]:
    questions: list[str] = []
    if filters.max_price is None:
        questions.append("What is the highest price you are comfortable with?")
    if filters.min_rating is None:
        questions.append("Should I enforce a minimum star rating?")
    if not filters.must_have:
        questions.append("Which features are non-negotiable?")
    return questions[:3]
