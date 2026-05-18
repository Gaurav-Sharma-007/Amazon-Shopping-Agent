import { useMemo, useState } from "react";
import {
  Bot,
  CheckCircle2,
  Database,
  Filter,
  Loader2,
  RotateCcw,
  Search,
  Send,
  SlidersHorizontal,
  Sparkles,
  Star,
} from "lucide-react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const emptyFilters = {
  query: "",
  min_price: "",
  max_price: "",
  min_rating: "",
  min_reviews: "",
  prime_only: false,
  brands: "",
  must_have: "",
  avoid: "",
  sort_goal: "best_match",
};

function normalizeFilters(filters) {
  return {
    query: filters.query.trim(),
    min_price: numberOrNull(filters.min_price),
    max_price: numberOrNull(filters.max_price),
    min_rating: numberOrNull(filters.min_rating),
    min_reviews: intOrNull(filters.min_reviews),
    prime_only: filters.prime_only,
    brands: splitTerms(filters.brands),
    must_have: splitTerms(filters.must_have),
    avoid: splitTerms(filters.avoid),
    sort_goal: filters.sort_goal,
  };
}

function numberOrNull(value) {
  if (value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function intOrNull(value) {
  if (value === "") return null;
  const number = Number.parseInt(value, 10);
  return Number.isFinite(number) ? number : null;
}

function splitTerms(value) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function App() {
  const [sessionId, setSessionId] = useState(
    () => localStorage.getItem("amazon-recommender-session") || null
  );
  const [message, setMessage] = useState("");
  const [filters, setFilters] = useState(emptyFilters);
  const [chat, setChat] = useState([]);
  const [recommendation, setRecommendation] = useState(null);
  const [trace, setTrace] = useState([]);
  const [productsSaved, setProductsSaved] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const topProduct = recommendation?.top_product;
  const rankedProducts = recommendation?.ranked_products || [];
  const activeFilterCount = useMemo(
    () =>
      Object.entries(filters).filter(([key, value]) => {
        if (key === "sort_goal") return value !== "best_match";
        if (typeof value === "boolean") return value;
        return String(value).trim().length > 0;
      }).length,
    [filters]
  );

  async function submit(event) {
    event.preventDefault();
    if (!message.trim() || loading) return;

    setLoading(true);
    setError("");
    const outgoing = message.trim();
    setChat((items) => [...items, { role: "user", content: outgoing }]);

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          message: outgoing,
          filters: normalizeFilters(filters),
        }),
      });
      if (!response.ok) {
        let detail = "";
        try {
          const payload = await response.json();
          detail = Array.isArray(payload.detail)
            ? payload.detail.map((item) => item.msg).join(", ")
            : payload.detail || "";
        } catch {
          detail = await response.text();
        }
        throw new Error(detail || `Request failed with ${response.status}`);
      }
      const payload = await response.json();
      setSessionId(payload.session_id);
      localStorage.setItem("amazon-recommender-session", payload.session_id);
      setChat(payload.messages);
      setRecommendation(payload.recommendation);
      setTrace(payload.trace);
      setProductsSaved(payload.products_saved);
      setMessage("");
    } catch (err) {
      const message =
        err instanceof TypeError
          ? `Could not reach the backend at ${API_URL}.`
          : err.message;
      setError(message);
      setChat((items) => [
        ...items,
        {
          role: "assistant",
          content: "I could not complete that recommendation request yet.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="filter-panel">
        <div className="panel-heading">
          <div className="panel-title">
            <SlidersHorizontal size={20} />
            <h1>Optional Filters</h1>
          </div>
          <button
            className="icon-button"
            type="button"
            title="Clear filters"
            onClick={() => setFilters(emptyFilters)}
          >
            <RotateCcw size={17} />
          </button>
        </div>

        <label>
          <span>Keyword Override</span>
          <input
            value={filters.query}
            onChange={(event) => setFilters({ ...filters, query: event.target.value })}
            placeholder="Optional: wireless headphones"
          />
        </label>

        <div className="field-grid">
          <label>
            <span>Min Price</span>
            <input
              type="number"
              value={filters.min_price}
              onChange={(event) => setFilters({ ...filters, min_price: event.target.value })}
              min="0"
            />
          </label>
          <label>
            <span>Max Price</span>
            <input
              type="number"
              value={filters.max_price}
              onChange={(event) => setFilters({ ...filters, max_price: event.target.value })}
              min="0"
            />
          </label>
        </div>

        <div className="field-grid">
          <label>
            <span>Min Rating</span>
            <input
              type="number"
              value={filters.min_rating}
              onChange={(event) => setFilters({ ...filters, min_rating: event.target.value })}
              min="0"
              max="5"
              step="0.1"
            />
          </label>
          <label>
            <span>Min Reviews</span>
            <input
              type="number"
              value={filters.min_reviews}
              onChange={(event) => setFilters({ ...filters, min_reviews: event.target.value })}
              min="0"
            />
          </label>
        </div>

        <label>
          <span>Brands</span>
          <input
            value={filters.brands}
            onChange={(event) => setFilters({ ...filters, brands: event.target.value })}
            placeholder="Sony, Anker, Apple"
          />
        </label>

        <label>
          <span>Must Have</span>
          <input
            value={filters.must_have}
            onChange={(event) => setFilters({ ...filters, must_have: event.target.value })}
            placeholder="noise cancellation, USB-C"
          />
        </label>

        <label>
          <span>Avoid</span>
          <input
            value={filters.avoid}
            onChange={(event) => setFilters({ ...filters, avoid: event.target.value })}
            placeholder="refurbished, wired"
          />
        </label>

        <div className="segmented" role="group" aria-label="Ranking goal">
          {["best_match", "value", "rating", "budget"].map((goal) => (
            <button
              key={goal}
              className={filters.sort_goal === goal ? "active" : ""}
              type="button"
              onClick={() => setFilters({ ...filters, sort_goal: goal })}
            >
              {goal.replace("_", " ")}
            </button>
          ))}
        </div>

        <label className="toggle-row">
          <input
            type="checkbox"
            checked={filters.prime_only}
            onChange={(event) => setFilters({ ...filters, prime_only: event.target.checked })}
          />
          <span>Prime only</span>
        </label>

        <div className="metric-row">
          <div>
            <Filter size={18} />
            <strong>{activeFilterCount}</strong>
            <span>active filters</span>
          </div>
          <div>
            <Database size={18} />
            <strong>{productsSaved}</strong>
            <span>saved</span>
          </div>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p>LangGraph shopping agent</p>
            <h2>Find the product that fits the conversation.</h2>
          </div>
          <div className="status-pill">
            <CheckCircle2 size={16} />
            Context memory on
          </div>
        </header>

        <div className="content-grid">
          <section className="chat-panel">
            <div className="chat-log">
              {chat.length === 0 && (
                <div className="empty-state">
                  <Bot size={36} />
                  <p>Describe what you want. Add filters only when they matter.</p>
                </div>
              )}
              {chat.map((item, index) => (
                <article key={`${item.role}-${index}`} className={`message ${item.role}`}>
                  <span>{item.role === "user" ? "You" : "AI"}</span>
                  <p>{item.content}</p>
                </article>
              ))}
            </div>

            <form className="composer" onSubmit={submit}>
              <Search size={18} />
              <input
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                placeholder="Describe a product, or include things like brand, rating, reviews, and price"
              />
              <button type="submit" disabled={loading || !message.trim()} title="Send">
                {loading ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
              </button>
            </form>
            {error && <p className="error">{error}</p>}
          </section>

          <section className="results-panel">
            <div className="section-heading">
              <Sparkles size={20} />
              <h3>Recommendation</h3>
            </div>

            {topProduct ? (
              <ProductCard product={topProduct} featured />
            ) : (
              <div className="placeholder">
                <Star size={28} />
                <p>Your ranked recommendation will appear here.</p>
              </div>
            )}

            {recommendation?.explanation?.length > 0 && (
              <div className="explanation">
                {recommendation.explanation.map((item, index) => (
                  <p key={index}>{item}</p>
                ))}
              </div>
            )}

            <div className="product-list">
              {rankedProducts.slice(1, 5).map((product) => (
                <ProductCard key={product.product_id} product={product} />
              ))}
            </div>

            {trace.length > 0 && (
              <div className="trace">
                {trace.map((item, index) => (
                  <span key={index}>{item}</span>
                ))}
              </div>
            )}
          </section>
        </div>
      </section>
    </main>
  );
}

function ProductCard({ product, featured = false }) {
  return (
    <article className={`product-card ${featured ? "featured" : ""}`}>
      <div className="product-image">
        {product.image_url ? <img src={product.image_url} alt="" /> : <Sparkles size={30} />}
      </div>
      <div className="product-copy">
        <div className="product-title-row">
          <h4>{product.title}</h4>
          <strong>{Math.round(product.score)}</strong>
        </div>
        <div className="product-meta">
          {product.price != null && <span>${product.price.toFixed(2)}</span>}
          {product.rating != null && <span>{product.rating.toFixed(1)} stars</span>}
          {product.review_count != null && <span>{product.review_count.toLocaleString()} reviews</span>}
          {product.is_prime && <span>Prime</span>}
        </div>
        <div className="reason-list">
          {(product.reasons || []).slice(0, 3).map((reason, index) => (
            <p key={index}>{reason}</p>
          ))}
        </div>
        <a href={product.url} target="_blank" rel="noreferrer">
          Open listing
        </a>
      </div>
    </article>
  );
}

export default App;
