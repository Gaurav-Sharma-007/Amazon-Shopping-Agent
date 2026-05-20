from __future__ import annotations

"""
Product Recommendation Graph — v2
==================================
LangGraph pipeline with six nodes:

  orchestrator        — loads session memory, routes message
  intent_agent        — LLM Query Planner → SearchSpec  (+ regex fallback)
  scraping_agent      — Amazon SERP scraper with native URL filters + TTL cache
  relevance_filter    — LLM scores each product 0-10; drops irrelevant ones
  ranking_agent       — LLM ranks remaining products with reasoning
  recommendation_llm  — Bedrock generates final human-readable response
  database_agent      — persists RankedProducts to DynamoDB / local JSON

Key improvements over v1:
  - Intent is parsed by Bedrock Claude → SearchSpec (not regex).
  - All major filters (price, rating, Prime, brand, sort) are baked into the
    Amazon search URL BEFORE Playwright loads any page.
  - A dedicated Bedrock call scores relevance of each card and drops junk.
  - A second Bedrock call ranks the remaining products with explicit reasoning.
  - Result cache (DynamoDB or in-process) avoids redundant Playwright calls.
"""

import logging

from langgraph.graph import END, StateGraph

from .llm import RecommendationLLM
from .llm_query_planner import LLMQueryPlanner
from .llm_relevance_filter import LLMRelevanceFilter
from .llm_ranker import LLMRanker
from .marketplaces import marketplace_label
from .memory import SessionMemory
from .models import (
    AgentState,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Product,
    ProductFilters,
    RankedProduct,
    Recommendation,
    ScoredProduct,
    SearchSpec,
)
from .scraper import AmazonCatalogScraper
from .storage import ProductRepository, QueryCache, build_product_repository, build_query_cache

logger = logging.getLogger(__name__)


class ProductRecommendationGraph:
    def __init__(
        self,
        memory: SessionMemory | None = None,
        scraper: AmazonCatalogScraper | None = None,
        repository: ProductRepository | None = None,
        query_cache: QueryCache | None = None,
        llm: RecommendationLLM | None = None,
        planner: LLMQueryPlanner | None = None,
        relevance_filter: LLMRelevanceFilter | None = None,
        ranker: LLMRanker | None = None,
    ) -> None:
        self.memory = memory or SessionMemory()
        self.scraper = scraper or AmazonCatalogScraper()
        self.repository = repository or build_product_repository()
        self.query_cache = query_cache or build_query_cache()
        self.llm = llm or RecommendationLLM()
        self.planner = planner or LLMQueryPlanner()
        self.relevance_filter = relevance_filter or LLMRelevanceFilter()
        self.ranker = ranker or LLMRanker()
        self.graph = self._build_graph().compile()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, request: ChatRequest) -> ChatResponse:
        session_id = request.session_id or self.memory.create_session_id()
        previous = self.memory.load(session_id)
        messages = previous.get("messages", [])
        filters = ProductFilters(**previous.get("filters", {}))

        # Merge any explicit filters from the request
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
            ChatMessage(**msg) if isinstance(msg, dict) else msg
            for msg in result["messages"]
        ]
        final_messages.append(
            ChatMessage(role="assistant", content=recommendation.summary)
        )
        self.memory.save(
            session_id, final_messages, final_filters, recommendation.model_dump()
        )

        return ChatResponse(
            session_id=session_id,
            messages=final_messages,
            filters=final_filters,
            recommendation=recommendation,
            products_saved=result.get("products_saved", 0),
            trace=result.get("trace", []),
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("orchestrator", self._orchestrator)
        graph.add_node("intent_agent", self._intent_agent)
        graph.add_node("scraping_agent", self._scraping_agent)
        graph.add_node("relevance_filter", self._relevance_filter_node)
        graph.add_node("ranking_agent", self._ranking_agent)
        graph.add_node("database_agent", self._database_agent)
        graph.add_node("recommendation_llm", self._recommendation_llm)

        graph.set_entry_point("orchestrator")
        graph.add_edge("orchestrator", "intent_agent")
        graph.add_edge("intent_agent", "scraping_agent")
        graph.add_edge("scraping_agent", "relevance_filter")
        graph.add_edge("relevance_filter", "ranking_agent")
        graph.add_edge("ranking_agent", "database_agent")
        graph.add_edge("database_agent", "recommendation_llm")
        graph.add_edge("recommendation_llm", END)
        return graph

    # ------------------------------------------------------------------
    # Node: Orchestrator
    # ------------------------------------------------------------------

    async def _orchestrator(self, state: AgentState) -> AgentState:
        trace = state.get("trace", [])
        trace.append("Orchestrator: loaded session memory and routed to intent agent.")
        messages = state.get("messages", [])
        messages.append({"role": "user", "content": state["user_message"]})
        return {**state, "messages": messages, "trace": trace}

    # ------------------------------------------------------------------
    # Node: Intent Agent  (LLM Query Planner → SearchSpec)
    # ------------------------------------------------------------------

    async def _intent_agent(self, state: AgentState) -> AgentState:
        filters = ProductFilters(**state["filters"])
        spec: SearchSpec = await self.planner.plan(
            user_message=state["user_message"],
            session_messages=state.get("messages", []),
            current_filters=filters,
        )

        # Sync ProductFilters back from SearchSpec so session memory stays
        # compatible and the API surface is unchanged.
        url_f = spec.amazon_url_filters
        if url_f.p_36:
            parts = url_f.p_36.split("-")
            if len(parts) == 2:
                filters.min_price = int(parts[0]) / 100
                filters.max_price = int(parts[1]) / 100
        if url_f.p_72:
            # Store the rating indirectly — the node ID implies ≥4★ or ≥3★
            if url_f.p_72 in {"2421889011", "328520031", "669342031", "1292115031"}:
                filters.min_rating = 4.0
            else:
                filters.min_rating = 3.0
        if url_f.p_85:
            filters.prime_only = True
        filters.brands = spec.brands
        filters.must_have = spec.must_have_keywords
        filters.avoid = spec.avoid_keywords
        filters.sort_goal = spec.sort_goal
        filters.query = spec.search_term
        filters.marketplace = spec.marketplace

        trace = state.get("trace", [])
        trace.append(
            f"Intent agent (LLM Query Planner): search_term='{spec.search_term}', "
            f"marketplace={marketplace_label(spec.marketplace)}, "
            f"url_filters={url_f.model_dump(exclude_none=True)}."
        )
        return {
            **state,
            "filters": filters.model_dump(),
            "scrape_plan": spec.model_dump(),
            "search_query": spec.search_term,
            "trace": trace,
        }

    # ------------------------------------------------------------------
    # Node: Scraping Agent
    # ------------------------------------------------------------------

    async def _scraping_agent(self, state: AgentState) -> AgentState:
        spec = SearchSpec(**state["scrape_plan"])
        trace = state.get("trace", [])
        cache_hit = False

        # Check persistent query cache first
        from .scraper import _cache_key
        from .config import get_settings
        key = _cache_key(spec)
        cached = self.query_cache.get(key)
        if cached:
            products = cached
            cache_hit = True
            trace.append(
                f"Scraping agent: cache hit ({len(products)} products, key={key[:8]}…)."
            )
        else:
            products = await self.scraper.search(spec)
            settings = get_settings()
            self.query_cache.put(key, products, settings.query_cache_ttl_seconds)
            trace.append(
                f"Scraping agent: scraped {len(products)} products from Amazon "
                f"with native URL filters."
            )

        return {
            **state,
            "products": [p.model_dump() for p in products],
            "cache_hit": cache_hit,
            "trace": trace,
        }

    # ------------------------------------------------------------------
    # Node: Relevance Filter  (LLM scores each product 0-10)
    # ------------------------------------------------------------------

    async def _relevance_filter_node(self, state: AgentState) -> AgentState:
        spec = SearchSpec(**state["scrape_plan"])
        products = [Product(**p) for p in state.get("products", [])]
        scored = await self.relevance_filter.filter(products, spec)
        trace = state.get("trace", [])
        dropped = len(products) - len(scored)
        trace.append(
            f"Relevance filter: {len(scored)} products kept, "
            f"{dropped} dropped as irrelevant."
        )
        return {
            **state,
            "scored_products": [p.model_dump() for p in scored],
            "trace": trace,
        }

    # ------------------------------------------------------------------
    # Node: Ranking Agent  (LLM ranks with value/intent/quality axes)
    # ------------------------------------------------------------------

    async def _ranking_agent(self, state: AgentState) -> AgentState:
        spec = SearchSpec(**state["scrape_plan"])
        filters = ProductFilters(**state["filters"])
        scored = [ScoredProduct(**p) for p in state.get("scored_products", [])]
        ranked = await self.ranker.rank(scored, spec, filters)
        trace = state.get("trace", [])
        trace.append(
            f"Ranking agent (LLM): ranked {len(ranked)} products by "
            "value_for_money, user_intent_fit, quality_signals."
        )
        return {
            **state,
            "ranked_products": [p.model_dump() for p in ranked],
            "trace": trace,
        }

    # ------------------------------------------------------------------
    # Node: Database Agent
    # ------------------------------------------------------------------

    async def _database_agent(self, state: AgentState) -> AgentState:
        ranked = [RankedProduct(**p) for p in state.get("ranked_products", [])]
        saved = await self.repository.save_products(state["session_id"], ranked)
        trace = state.get("trace", [])
        trace.append(f"Database agent: saved {saved} ranked products to persistent store.")
        return {**state, "products_saved": saved, "trace": trace}

    # ------------------------------------------------------------------
    # Node: Recommendation LLM  (response builder)
    # ------------------------------------------------------------------

    async def _recommendation_llm(self, state: AgentState) -> AgentState:
        filters = ProductFilters(**state["filters"])
        ranked = [RankedProduct(**p) for p in state.get("ranked_products", [])]
        recommendation = await self.llm.explain(filters, ranked)
        trace = state.get("trace", [])
        trace.append(self.llm.last_status)
        return {**state, "recommendation": recommendation.model_dump(), "trace": trace}
