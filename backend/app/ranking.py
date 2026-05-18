from __future__ import annotations

import math

from .models import Product, ProductFilters, RankedProduct


def rank_products(products: list[Product], filters: ProductFilters) -> list[RankedProduct]:
    if not products:
        return []

    prices = [product.price for product in products if product.price is not None]
    min_price = min(prices) if prices else 0
    max_price = max(prices) if prices else 1
    spread = max(max_price - min_price, 1)

    ranked: list[RankedProduct] = []
    for product in products:
        rating_score = ((product.rating or 3.5) / 5) * 30
        review_score = min(math.log10((product.review_count or 1) + 1) / 5, 1) * 20
        price_score = _price_score(product.price, min_price, spread, filters.sort_goal)
        preference_score, reasons = _preference_score(product, filters)
        prime_score = 8 if product.is_prime else 0

        breakdown = {
            "rating": round(rating_score, 2),
            "reviews": round(review_score, 2),
            "price": round(price_score, 2),
            "preferences": round(preference_score, 2),
            "prime": round(prime_score, 2),
        }
        total = round(sum(breakdown.values()), 2)
        reasons.extend(_default_reasons(product, filters))

        ranked.append(
            RankedProduct(
                **product.model_dump(),
                score=total,
                score_breakdown=breakdown,
                reasons=reasons[:4],
            )
        )

    return sorted(ranked, key=lambda item: item.score, reverse=True)


def _price_score(price: float | None, min_price: float, spread: float, goal: str) -> float:
    if price is None:
        return 10
    lower_is_better = 1 - ((price - min_price) / spread)
    if goal == "budget":
        return lower_is_better * 30
    if goal == "value":
        return lower_is_better * 22
    return lower_is_better * 14


def _preference_score(product: Product, filters: ProductFilters) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    title = product.title.lower()

    for term in filters.must_have:
        if _matches_term(term, title):
            score += 8
            reasons.append(f"Matches must-have preference: {term}")

    if filters.brands and product.brand:
        if product.brand.lower() in {brand.lower() for brand in filters.brands}:
            score += 10
            reasons.append(f"Preferred brand match: {product.brand}")

    return min(score, 28), reasons


def _matches_term(term: str, title: str) -> bool:
    normalized_term = _normalize_text(term)
    normalized_title = _normalize_text(title)
    if normalized_term in normalized_title:
        return True
    return all(token in normalized_title for token in normalized_term.split())


def _normalize_text(value: str) -> str:
    return (
        value.lower()
        .replace("cancelling", "cancel")
        .replace("cancellation", "cancel")
        .replace("canceling", "cancel")
        .replace("-", " ")
    )


def _default_reasons(product: Product, filters: ProductFilters) -> list[str]:
    reasons: list[str] = []
    if product.rating:
        reasons.append(f"{product.rating:.1f}/5 rating supports quality confidence")
    if product.review_count:
        reasons.append(f"{product.review_count:,} reviews give useful signal")
    if product.price is not None:
        if filters.max_price and product.price <= filters.max_price:
            reasons.append(f"Inside your budget at ${product.price:.2f}")
        else:
            reasons.append(f"Listed around ${product.price:.2f}")
    if product.is_prime:
        reasons.append("Prime indicator found in listing")
    return reasons
