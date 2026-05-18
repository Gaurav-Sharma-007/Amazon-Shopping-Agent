from __future__ import annotations

import re

from langgraph.graph import END, StateGraph

from .llm import RecommendationLLM
from .memory import SessionMemory
from .marketplaces import (
    find_marketplace_in_text,
    marketplace_label,
    strip_marketplace_terms,
)
from .models import (
    AgentState,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Product,
    ProductFilters,
    RankedProduct,
    Recommendation,
)
from .ranking import rank_products
from .scraper import AmazonCatalogScraper
from .storage import ProductRepository, build_product_repository


PRICE_UNDER_RE = re.compile(
    r"(?:under|below|less than|max|budget)\s*(?:[$₹]|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)(?![\d.])(?!\s*(?:stars?|reviews?|ratings?))",
    re.I,
)
PRICE_OVER_RE = re.compile(
    r"(?:over|above|at least|min(?:imum)?)\s*(?:[$₹]|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)(?![\d.])(?!\s*(?:stars?|reviews?|ratings?))",
    re.I,
)
PRICE_RANGE_RE = re.compile(
    r"(?:price\s*)?(?:range\s*)?(?:of|between|from)?\s*(?:[$₹]|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)\s*(?:to|-|and)\s*(?:[$₹]|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)",
    re.I,
)
RATING_RE = re.compile(r"(\d(?:\.\d)?)\s*(?:star|stars|\+)", re.I)
RATING_KEYWORD_RE = re.compile(
    r"(?:min(?:imum)?\s*)?(?:rating|rated)\s*(?:of|at least|above|over)?\s*(\d(?:\.\d)?)",
    re.I,
)
REVIEWS_RE = re.compile(
    r"(?:(?:at least|min(?:imum)?|over|above)\s*)?(\d[\d,]*)\+?\s*(?:reviews?|ratings?)",
    re.I,
)
COLOR_WORDS = {
    "black",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "pink",
    "purple",
    "violet",
    "orange",
    "brown",
    "grey",
    "gray",
    "silver",
    "gold",
    "golden",
    "beige",
    "cream",
    "transparent",
}
REFINEMENT_RE = re.compile(
    r"\b(change|switch|make|same|instead|only|just|update|modify|replace)\b",
    re.I,
)


class ProductRecommendationGraph:
    def __init__(
        self,
        memory: SessionMemory | None = None,
        scraper: AmazonCatalogScraper | None = None,
        repository: ProductRepository | None = None,
        llm: RecommendationLLM | None = None,
    ) -> None:
        self.memory = memory or SessionMemory()
        self.scraper = scraper or AmazonCatalogScraper()
        self.repository = repository or build_product_repository()
        self.llm = llm or RecommendationLLM()
        self.graph = self._build_graph().compile()

    async def run(self, request: ChatRequest) -> ChatResponse:
        session_id = request.session_id or self.memory.create_session_id()
        previous = self.memory.load(session_id)
        messages = previous.get("messages", [])
        filters = ProductFilters(**previous.get("filters", {}))

        incoming = request.filters.model_dump(exclude_unset=True)
        if incoming:
            filters = filters.model_copy(update=incoming)

        state: AgentState = {
            "session_id": session_id,
            "user_message": request.message,
            "messages": messages,
            "filters": filters.model_dump(),
            "trace": [],
        }

        result = await self.graph.ainvoke(state)
        final_filters = ProductFilters(**result["filters"])
        recommendation = Recommendation(**result["recommendation"])

        final_messages = [
            ChatMessage(**message) if isinstance(message, dict) else message
            for message in result["messages"]
        ]
        final_messages.append(ChatMessage(role="assistant", content=recommendation.summary))
        self.memory.save(
            session_id,
            final_messages,
            final_filters,
            recommendation.model_dump(),
        )

        return ChatResponse(
            session_id=session_id,
            messages=final_messages,
            filters=final_filters,
            recommendation=recommendation,
            products_saved=result.get("products_saved", 0),
            trace=result.get("trace", []),
        )

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("orchestrator", self._orchestrator)
        graph.add_node("intent_agent", self._intent_agent)
        graph.add_node("scraping_agent", self._scraping_agent)
        graph.add_node("ranking_agent", self._ranking_agent)
        graph.add_node("database_agent", self._database_agent)
        graph.add_node("recommendation_llm", self._recommendation_llm)

        graph.set_entry_point("orchestrator")
        graph.add_edge("orchestrator", "intent_agent")
        graph.add_edge("intent_agent", "scraping_agent")
        graph.add_edge("scraping_agent", "ranking_agent")
        graph.add_edge("ranking_agent", "database_agent")
        graph.add_edge("database_agent", "recommendation_llm")
        graph.add_edge("recommendation_llm", END)
        return graph

    async def _orchestrator(self, state: AgentState) -> AgentState:
        trace = state.get("trace", [])
        trace.append("Orchestrator loaded previous session memory and routed to intent agent.")
        messages = state.get("messages", [])
        messages.append({"role": "user", "content": state["user_message"]})
        return {**state, "messages": messages, "trace": trace}

    async def _intent_agent(self, state: AgentState) -> AgentState:
        filters = ProductFilters(**state["filters"])
        message = state["user_message"]
        lower = message.lower()

        range_match = PRICE_RANGE_RE.search(message)
        max_match = PRICE_UNDER_RE.search(message)
        min_match = PRICE_OVER_RE.search(message)
        rating_match = RATING_RE.search(message) or RATING_KEYWORD_RE.search(message)
        reviews_match = REVIEWS_RE.search(message)
        marketplace = find_marketplace_in_text(message)
        if marketplace:
            filters.marketplace = marketplace
        if range_match:
            low = float(range_match.group(1).replace(",", ""))
            high = float(range_match.group(2).replace(",", ""))
            filters.min_price = min(low, high)
            filters.max_price = max(low, high)
        elif max_match:
            filters.max_price = float(max_match.group(1).replace(",", ""))
        if min_match and not range_match:
            filters.min_price = float(min_match.group(1).replace(",", ""))
        if rating_match:
            filters.min_rating = min(5, float(rating_match.group(1)))
        if reviews_match:
            filters.min_reviews = int(reviews_match.group(1).replace(",", ""))
        if "prime" in lower:
            filters.prime_only = True
        if any(word in lower for word in ["cheap", "budget", "affordable"]):
            filters.sort_goal = "budget"
        if any(word in lower for word in ["value", "best for money", "worth"]):
            filters.sort_goal = "value"

        colors = _extract_colors(message)
        if colors:
            filters.must_have = [
                item for item in filters.must_have if item.lower() not in COLOR_WORDS
            ]
            for color in colors:
                if color not in filters.must_have:
                    filters.must_have.append(color)

        extracted_query = _extract_query(message)
        if extracted_query and not _is_refinement_message(message, extracted_query, filters.query):
            filters.query = extracted_query

        must_have = _extract_after_keywords(
            lower, ["with ", "must have ", "needs to have ", "that has "]
        )
        for term in must_have:
            if term not in filters.must_have and len(term) <= 40:
                filters.must_have.append(term)

        for brand in _extract_brands(message):
            if brand.lower() not in {item.lower() for item in filters.brands}:
                filters.brands.append(brand)

        for term in _extract_after_keywords(lower, ["avoid ", "without ", "exclude "]):
            if term not in filters.avoid and len(term) <= 40:
                filters.avoid.append(term)

        trace = state.get("trace", [])
        trace.append(
            "Intent agent converted natural language into query, filters, "
            f"preferences, and marketplace: {marketplace_label(filters.marketplace)}."
        )
        return {**state, "filters": filters.model_dump(), "search_query": filters.query, "trace": trace}

    async def _scraping_agent(self, state: AgentState) -> AgentState:
        filters = ProductFilters(**state["filters"])
        products = await self.scraper.search(filters)
        trace = state.get("trace", [])
        trace.append(f"Scraping agent collected {len(products)} catalog candidates.")
        return {**state, "products": [product.model_dump() for product in products], "trace": trace}

    async def _ranking_agent(self, state: AgentState) -> AgentState:
        filters = ProductFilters(**state["filters"])
        products = [Product(**product) for product in state.get("products", [])]
        ranked = rank_products(products, filters)
        trace = state.get("trace", [])
        trace.append("Ranking agent scored candidates using quality, price, delivery, and preference fit.")
        return {
            **state,
            "ranked_products": [product.model_dump() for product in ranked],
            "trace": trace,
        }

    async def _database_agent(self, state: AgentState) -> AgentState:
        ranked = [RankedProduct(**product) for product in state.get("ranked_products", [])]
        saved = await self.repository.save_products(state["session_id"], ranked)
        trace = state.get("trace", [])
        trace.append(f"Database agent saved {saved} ranked products for the session.")
        return {**state, "products_saved": saved, "trace": trace}

    async def _recommendation_llm(self, state: AgentState) -> AgentState:
        filters = ProductFilters(**state["filters"])
        ranked = [RankedProduct(**product) for product in state.get("ranked_products", [])]
        recommendation = await self.llm.explain(filters, ranked)
        trace = state.get("trace", [])
        trace.append(self.llm.last_status)
        return {**state, "recommendation": recommendation.model_dump(), "trace": trace}


def _extract_query(message: str) -> str:
    cleaned = strip_marketplace_terms(message)
    cleaned = PRICE_RANGE_RE.sub("", cleaned)
    cleaned = PRICE_UNDER_RE.sub("", cleaned)
    cleaned = PRICE_OVER_RE.sub("", cleaned)
    cleaned = re.sub(r"\b(?:in|under|un)\s+the\s+price\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\bprice\s+range\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\bun\s+the\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\b\d(?:\.\d)?\s*(star|stars|\+)\b", "", cleaned, flags=re.I)
    cleaned = RATING_KEYWORD_RE.sub("", cleaned)
    cleaned = REVIEWS_RE.sub("", cleaned)
    cleaned = re.sub(r"\bbrands?\s*(?:are|include|:|=)?\s*[^.;,]+(?:[,;]\s*[^.;,]+)*", "", cleaned, flags=re.I)
    cleaned = re.sub(
        r"\bfrom\s+[a-z0-9&\-\s,]+?(?=\s+\b(?:with|under|below|less than|over|above|at least|avoid|without|exclude|rating|reviews?|prime)\b|$)",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"\b(i want|i need|find me|recommend|show me|looking for|best|amazon|prime)\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\b(change|switch|make|update|modify|replace)\s+(the\s+)?colou?r\s+(to\s+)?\w+\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\b(colou?r)\s+(to\s+)?\w+\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\b(colou?red|color)\b", "", cleaned, flags=re.I)
    for color in COLOR_WORDS:
        cleaned = re.sub(rf"\b{re.escape(color)}\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\b(with|must have|avoid|without|exclude)\b.*", "", cleaned, flags=re.I)
    cleaned = " ".join(cleaned.split())
    return cleaned.strip(" ,.-")[:120]


def _extract_after_keywords(message: str, keywords: list[str]) -> list[str]:
    results: list[str] = []
    for keyword in keywords:
        if keyword in message:
            tail = message.split(keyword, 1)[1]
            tail = re.split(
                r"\b(?:under|below|less than|over|above|at least|for|and avoid|minimum|rating|reviews?|brands?)\b",
                tail,
            )[0]
            for part in re.split(r",| and ", tail):
                normalized = part.strip(" .")
                if normalized and normalized not in {"prime", "amazon prime"} and len(normalized.split()) <= 4:
                    results.append(normalized)
    return results[:5]


def _extract_brands(message: str) -> list[str]:
    match = re.search(
        r"\bbrands?\s*(?:are|include|:|=)?\s*([a-z0-9][a-z0-9&\-\s,]+)",
        message,
        re.I,
    )
    if not match:
        match = re.search(
            r"\bfrom\s+([a-z0-9][a-z0-9&\-\s,]+)",
            message,
            re.I,
        )
    if not match:
        return []

    tail = re.split(
        r"\b(?:under|below|less than|over|above|at least|with|must have|avoid|without|exclude|rating|reviews?|prime)\b",
        match.group(1),
        maxsplit=1,
        flags=re.I,
    )[0]
    brands: list[str] = []
    for part in re.split(r",|/|\bor\b|\band\b", tail, flags=re.I):
        brand = " ".join(part.strip(" .:-").split())
        if brand and len(brand.split()) <= 3:
            brands.append(brand.upper() if brand.isupper() else brand.title())
    return brands[:5]


def _extract_colors(message: str) -> list[str]:
    lowered = message.lower()
    colors = [color for color in COLOR_WORDS if re.search(rf"\b{re.escape(color)}\b", lowered)]
    return sorted(colors, key=lambda color: lowered.index(color))[:3]


def _is_refinement_message(message: str, extracted_query: str, current_query: str) -> bool:
    if not current_query:
        return False
    lowered_query = extracted_query.lower()
    if REFINEMENT_RE.search(message):
        return True
    if lowered_query in COLOR_WORDS:
        return True
    non_product_terms = COLOR_WORDS | {"color", "colour", "colored", "coloured"}
    return bool(lowered_query) and all(token in non_product_terms for token in lowered_query.split())
