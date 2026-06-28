# Statline

Ask plain-English questions about football and NBA data. Get SQL, answers, and charts — instantly.

Two Streamlit apps backed by a LangChain ReAct agent and DeepSeek V3. Type a question, the agent writes the SQL, executes it against a local SQLite database, and renders the result as a table and Vega-Lite chart — all in a single UI.

---

## Apps

| App | Dataset | Port |
|-----|---------|------|
| Statline · Football | 11 European leagues · 25,979 matches · 2008–2016 | 8501 |
| Statline · Hoops | NBA · 60,192 games · 4,815 players · 1946–2023 | 8502 |

Both apps share the same architecture and are launched together with one command.

---

## Features

- **Natural language → SQL → chart** in one step
- **Three-tier routing** — keyword triggers → LLM classifier → execution path:
  - Single-shot for straightforward aggregations
  - Entity lookup (DB-backed, no extra LLM call) for questions with team/player names
  - ReAct agent loop for multi-step and time-comparison queries
- **Enriched datasets** — Transfermarkt market values and per-match appearances (football); player salary history 2000–2025 (NBA)
- **Cost and token transparency** — every answer shows route, latency, token count, and cost
- **Read-only database** — LLM-generated SQL can never mutate or drop data
- **LangSmith tracing** — optional, full prompt/response visibility per query

---

## Architecture

```
question
   │
   ▼
route_question()          keyword regex → LLM classifier
   │
   ├── SIMPLE   ──► single_shot()     one LLM call → SQL → execute
   ├── ENTITY   ──► resolve_entities() + single_shot()
   └── COMPLEX  ──► react()           LangChain ReAct agent loop
                         │
                         ▼
                   get_chart_spec()   Vega-Lite JSON from LLM
                         │
                         ▼
                   render_result()    table + chart + SQL + cost badge
```

`utils.py` is the shared core — all routing, execution, theming, and cost logic lives there. Each app's `app.py` holds only dataset-specific config: DB path, schema hints, entity tables, and sample questions.

---

## Datasets

Download these from Kaggle and place them as described below.

**Football** — [European Soccer Database](https://www.kaggle.com/datasets/hugomathien/soccer)
- Place `database.sqlite` in `Football Agent/`
- Run `python Football Agent/load_football_enrichment.py` to add Transfermarkt enrichment

**NBA** — [NBA Database](https://www.kaggle.com/datasets/wyattowalsh/basketball)
- Place `nba.sqlite` in `Nba Agent 2/NBA Data/`
- Run `python Nba Agent 2/load_salary.py` to add salary data

The `.sqlite` files are gitignored — you must download them yourself.

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/melgwyncarmichael-a11y/statline.git
cd statline
```

**2. Create a virtual environment and install dependencies**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Configure environment variables**
```bash
cp .env.example .env
# Edit .env and add your DEEPSEEK_API_KEY
```

**4. Download the datasets** (see Datasets section above)

**5. Launch both apps**
```bash
chmod +x launch.command
./launch.command
```

Both apps open in your browser. The terminal is free after launch — the apps run as background processes.

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DEEPSEEK_API_KEY` | Yes | DeepSeek API key — [platform.deepseek.com](https://platform.deepseek.com) |
| `LANGSMITH_TRACING` | No | Set to `true` to enable LangSmith tracing |
| `LANGSMITH_ENDPOINT` | No | LangSmith API endpoint |
| `LANGSMITH_API_KEY` | No | LangSmith API key — [smith.langchain.com](https://smith.langchain.com) |

---

## Stack

- [Streamlit](https://streamlit.io) — UI
- [LangChain](https://python.langchain.com) — agent framework and SQL toolkit
- [DeepSeek V3](https://platform.deepseek.com) — LLM for routing, SQL generation, and chart specs
- [SQLAlchemy](https://www.sqlalchemy.org) — database engine (read-only URI)
- [SQLite](https://www.sqlite.org) — local database, no server needed
- [LangSmith](https://smith.langchain.com) — optional observability and tracing

---

## Project structure

```
statline/
├── utils.py                        # shared routing, execution, theming, cost
├── launch.command                  # starts both apps, kills port conflicts
├── requirements.txt                # pinned dependencies
├── test_questions.py               # headless regression runner (22 questions)
├── Football Agent/
│   ├── app.py                      # football-specific config and UI
│   ├── load_football_enrichment.py # loads Transfermarkt data into SQLite
│   └── .streamlit/config.toml
└── Nba Agent 2/
    ├── app.py                      # NBA-specific config and UI
    ├── load_salary.py              # loads salary data into SQLite
    └── .streamlit/config.toml
```
