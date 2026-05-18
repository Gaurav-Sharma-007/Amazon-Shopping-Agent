# Amazon Preference Recommender

AI-assisted product narrowing tool with a React chat UI, filters, a LangGraph agent pipeline, Playwright catalog traversal, ranking, explainable recommendations, and DynamoDB-ready persistence.

## Architecture

Frontend UI:
- React + Vite chat workspace
- Required Amazon region selector plus structured filters for price, rating, reviews, brand, Prime, must-have, avoid, and ranking goal

Backend:
- FastAPI exposes `POST /api/chat`
- LangGraph orchestrates:
  1. Orchestrator agent loads conversation state
  2. Intent agent converts preference NLP into filters
  3. Scraping agent traverses public Amazon search pages with Playwright
  4. Ranking agent scores product candidates
  5. Database agent stores ranked products in DynamoDB or local JSON
  6. Recommendation LLM explains the result using Bedrock when available

State:
- Session memory is stored in `data/sessions.json`
- Product rows are stored in DynamoDB when enabled, otherwise `data/products.json`

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

```bash
cd frontend
npm install
```

Copy the example config if you want to change defaults:

```bash
cp config.example.ini config.ini
```

## Run

Backend:

```bash
source venv/bin/activate
python main.py
```

Frontend:

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173`.

## AWS Notes

To use DynamoDB, set this in `config.ini`:

```ini
[aws]
use_dynamodb = true
dynamodb_table = amazon-product-recommendations
```

Create a table with:
- partition key: `pk` string
- sort key: `sk` string

The app uses your normal AWS credential chain for both DynamoDB and Bedrock.

## Scraping Notes

The Playwright scraper only reads public search result pages for the selected Amazon region. It does not log in, bypass captcha, or evade access controls. In development, if Playwright or Amazon access fails, the backend returns deterministic sample products so the rest of the agent, ranking, memory, and UI flow remains testable.
