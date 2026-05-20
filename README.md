# Amazon Shopping Agent

AI-powered product recommender with a React chat UI, a 7-node LangGraph agent pipeline,
LLM-driven intent parsing, native Amazon SERP filter URLs, intelligent relevance filtering,
LLM-based ranking, and DynamoDB-backed persistence with result caching.

---

## Architecture

### Pipeline (7-node LangGraph graph)

```
User message
    │
    ▼  [Orchestrator]       Loads session memory, appends message
    ▼  [Intent Agent]       Bedrock LLM → SearchSpec JSON (regex fallback)
    ▼  [Scraping Agent]     Playwright + native Amazon URL filters + TTL cache
    ▼  [Relevance Filter]   Bedrock LLM scores each product 0-10, drops irrelevant ones
    ▼  [Ranking Agent]      Bedrock LLM ranks by value/intent fit/quality with reasoning
    ▼  [Database Agent]     Saves ranked products to DynamoDB (or local JSON)
    ▼  [Recommendation LLM] Bedrock generates human-readable summary + follow-ups
```

### Key Design Decisions

- **Filters applied on Amazon's page** — price, rating, Prime, brand and sort are encoded
  directly in the Amazon SERP URL (`p_36`, `p_72`, `p_85`, `p_89`, `s`) before Playwright
  loads any results. No post-scrape Python filtering.
- **LLM at every intelligence layer** — intent parsing, relevance scoring and ranking all
  use Bedrock. Every LLM call has a deterministic fallback so the pipeline never crashes
  when Bedrock is unreachable.
- **Result cache** — identical search specs are served from DynamoDB (or an in-process
  dict) for a configurable TTL, avoiding redundant Playwright round-trips.

### File Map

```
backend/app/
├── agents.py               LangGraph pipeline (all 7 nodes)
├── llm_query_planner.py    LLM Intent Agent  →  SearchSpec JSON
├── llm_relevance_filter.py LLM relevance scoring per product (0-10)
├── llm_ranker.py           LLM ranking with value/intent/quality axes
├── llm.py                  Recommendation LLM (response builder)
├── scraper.py              Playwright scraper + URL builder
├── models.py               Pydantic models (SearchSpec, AmazonURLFilters …)
├── storage.py              DynamoDB + local JSON + QueryCache
├── memory.py               Session memory (JSON-backed)
├── marketplaces.py         Amazon domain/currency/rating-node lookup
├── config.py               Settings loaded from config.ini / env vars
└── api.py                  FastAPI  POST /api/chat
frontend/src/
├── App.jsx                 React chat workspace
└── styles.css              Styles
```

---

## Setup

### Backend

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### Frontend

```bash
cd frontend
npm install
```

### Config

```bash
cp config.example.ini config.ini
# edit config.ini with your AWS region, model IDs, etc.
```

---

## Run

```bash
# Terminal 1 — backend
source venv/bin/activate
python main.py

# Terminal 2 — frontend
cd frontend
npm run dev
```

Open `http://localhost:5173`.

---

## AWS & Bedrock Configuration

The app uses your normal AWS credential chain (env vars, `~/.aws/credentials`, IAM role).
No credentials go in `config.ini`.

### Required IAM permissions

```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:InvokeModel",
    "bedrock:Converse",
    "dynamodb:BatchWriteItem",
    "dynamodb:GetItem",
    "dynamodb:PutItem",
    "dynamodb:Query"
  ],
  "Resource": "*"
}
```

### DynamoDB table

Create a table with:
- Partition key: `pk` (String)
- Sort key: `sk` (String)
- TTL attribute name: `ttl` (enables automatic cache expiry)

Enable in `config.ini`:

```ini
[aws]
use_dynamodb = true
dynamodb_table = your-table-name
```

---

## Changing the LLM Model

There are **two independent model slots** in `config.ini`:

| Key | Used by | Default |
|---|---|---|
| `query_planner_model_id` | Intent parser · Relevance filter · Ranker | Claude Haiku |
| `bedrock_model_id` | Final recommendation summary | Claude Haiku |

You can point each slot at any model available in your AWS account.

### Model ID format

```
<scope>.<provider>.<model-name>:<version>
```

- `us.`     — US cross-region inference profile (routes via `us-east-1` / `us-west-2`)
- `global.` — Global cross-region inference profile (requires additional AWS approval form)

---

### Available models on Amazon Bedrock

#### Anthropic Claude (recommended)

| Model | ID |
|---|---|
| Claude Haiku 4.5 (fast, cheap) | `us.anthropic.claude-haiku-4-5-20251001-v1:0` |
| Claude Sonnet 4.5 (balanced) | `us.anthropic.claude-sonnet-4-5-20251001-v1:0` |
| Claude Opus 4 (most capable) | `us.anthropic.claude-opus-4-20250514-v1:0` |

> To use Claude models, go to **AWS Console → Bedrock → Model Access** in `us-east-1`
> and request access to the Anthropic model family.

**Example — use Sonnet for planning/ranking, Haiku for responses:**

```ini
[aws]
query_planner_model_id = us.anthropic.claude-sonnet-4-5-20251001-v1:0
bedrock_model_id       = us.anthropic.claude-haiku-4-5-20251001-v1:0
```

---

#### Mistral AI (on Bedrock)

Mistral models are available natively on Amazon Bedrock. They follow a similar
`converse` API so they work as drop-in replacements.

| Model | ID |
|---|---|
| Mistral Large 2 | `mistral.mistral-large-2402-v1:0` |
| Mistral Small | `mistral.mistral-small-2402-v1:0` |
| Mixtral 8x7B | `mistral.mixtral-8x7b-instruct-v0:1` |

**Example — Mistral Large for planning, Haiku for responses:**

```ini
[aws]
query_planner_model_id = mistral.mistral-large-2402-v1:0
bedrock_model_id       = us.anthropic.claude-haiku-4-5-20251001-v1:0
```

> Enable Mistral models at **AWS Console → Bedrock → Model Access** in your region.
> Mistral is not available in all regions — `us-east-1` and `eu-west-1` are safest.

---

#### Amazon Titan / Nova (no approval needed)

Amazon's own models are enabled by default on every Bedrock account.

| Model | ID |
|---|---|
| Nova Micro (ultra-fast) | `us.amazon.nova-micro-v1:0` |
| Nova Lite | `us.amazon.nova-lite-v1:0` |
| Nova Pro | `us.amazon.nova-pro-v1:0` |

**Example — Nova for everything (zero approval required):**

```ini
[aws]
query_planner_model_id = us.amazon.nova-lite-v1:0
bedrock_model_id       = us.amazon.nova-micro-v1:0
```

---

#### Meta Llama (on Bedrock)

| Model | ID |
|---|---|
| Llama 3.3 70B Instruct | `us.meta.llama3-3-70b-instruct-v1:0` |
| Llama 3.1 8B Instruct | `us.meta.llama3-1-8b-instruct-v1:0` |

```ini
[aws]
query_planner_model_id = us.meta.llama3-3-70b-instruct-v1:0
bedrock_model_id       = us.meta.llama3-1-8b-instruct-v1:0
```

---

#### OpenAI (GPT-4o, GPT-4 Turbo) — via Bedrock Marketplace

OpenAI models are **not natively available** on Bedrock. To use GPT models you have
two options:

**Option A — AWS Marketplace (preview)**
AWS offers select OpenAI-compatible models via Bedrock Marketplace in some regions.
Check **AWS Console → Bedrock → Marketplace** for current availability.

**Option B — Call OpenAI directly (custom integration)**

To bypass Bedrock entirely and call the OpenAI API, replace the Bedrock `client.converse()`
call in `llm_query_planner.py`, `llm_relevance_filter.py`, `llm_ranker.py`, and `llm.py`
with an OpenAI client call.

1. Install the OpenAI SDK:
   ```bash
   pip install openai
   ```

2. Add your key to the environment:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

3. In each LLM file, replace the `boto3` Bedrock call with:
   ```python
   from openai import AsyncOpenAI

   client = AsyncOpenAI()
   response = await client.chat.completions.create(
       model="gpt-4o",          # or "gpt-4-turbo", "gpt-3.5-turbo"
       messages=[
           {"role": "system", "content": SYSTEM_PROMPT},
           {"role": "user",   "content": your_payload_json},
       ],
       response_format={"type": "json_object"},  # enforces JSON output
   )
   raw_text = response.choices[0].message.content
   ```

The rest of the pipeline (JSON parsing, Pydantic validation, fallback logic) is
unchanged — only the HTTP call differs.

---

### How to check which models your account can access

```bash
aws bedrock list-foundation-models \
    --region us-east-1 \
    --query 'modelSummaries[].modelId' \
    --output table
```

Or open **AWS Console → Amazon Bedrock → Model access** in `us-east-1`.

---

## Bedrock Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ResourceNotFoundException: Model use case details` | Using `global.` prefix without approval | Switch to `us.` prefix or fill out the Anthropic use-case form |
| `AccessDeniedException` | Model not enabled for your account | Request access in Bedrock console → Model Access |
| `ValidationException: Provided list of item keys contains duplicates` | Fixed ✅ | Deduplication added in `storage.py` |
| `TypeError: Float types are not supported` | Fixed ✅ | `parse_float=Decimal` added in `storage.py` |
| LLM falls back to regex | Bedrock unreachable | Pipeline continues with deterministic fallback — check AWS credentials |

---

## Scraping Notes

The Playwright scraper reads only **public** Amazon search result pages for the selected
marketplace. It does not log in, bypass captcha, or circumvent access controls.

Filters (price, star rating, Prime, brand, sort) are encoded in the Amazon SERP URL itself
before any page is loaded — Amazon's own backend applies them server-side.

In development, if Playwright or Amazon access fails, the backend returns deterministic
sample products so the full agent, ranking, memory, and UI flow remains testable.
