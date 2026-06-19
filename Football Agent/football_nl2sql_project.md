# Football NL→SQL + Visualisation Agent
**Project memory document** — captures all architecture decisions, costs, and discussions.
Last updated: May 2026

---

## 1. Project Origin

Started from an IBM Watson / WatsonX course demonstrating a natural language SQL agent using:
- `ibm/granite-3-2-8b-instruct` as the LLM
- `WatsonxLLM` + `langchain_community` SQL toolkit
- MySQL (Chinook database)
- LangChain's `create_sql_agent` with `ZERO_SHOT_REACT_DESCRIPTION`

**Goal:** Rebuild the same pattern without IBM's stack, using open/cheaper tools, for a personal demo to post on social media and learn from.

---

## 2. Dataset Decision

**Chosen:** European Soccer Database (Kaggle)
- URL: https://www.kaggle.com/datasets/hugomathien/soccer
- ~300 MB, already packaged as a `.sqlite` file — zero import/setup needed
- 11 European leagues, 25,000+ matches, 10,000+ players, seasons 2008–2016

**Why it works well for this pattern:**
- SQLite = no server, no credentials, one file
- Rich enough schema for interesting multi-table queries
- Relatable topic for social media audience
- Queries feel "magical" in plain English

### Key Tables

| Table | Key columns |
|---|---|
| `Match` | season, home_team_api_id, away_team_api_id, home_team_goal, away_team_goal |
| `Player` | player_name, birthday, height, weight |
| `Player_Attributes` | player_api_id, overall_rating, potential, finishing, dribbling, sprint_speed |
| `Team` | team_long_name, team_short_name |
| `League` | name |

**Gotcha:** Some columns have cryptic names (e.g. `home_team_api_id`). If the agent gives odd answers, add a system prompt with schema hints.

---

## 3. Core Architecture — NL→SQL Agent

### How It Works (5 layers)

```
User question (plain English)
        ↓
LangChain SQL Agent
  ├── Schema inspector    → reads table structures
  ├── LLM                 → writes SQL from question + schema
  ├── Query executor      → runs SQL against DB (read-only)
  ├── Prompt builder      → schema + question → LLM prompt
  └── Error handler       → retries bad SQL, fallback
        ↓
Database (SQLite)
        ↓
Response synthesiser (LLM reads raw rows → English answer)
        ↓
Natural language answer
```

### Why Token Count Is Higher Than Expected

The ReAct loop makes 2–4 API calls per question. Each call re-sends the full conversation history (LLMs are stateless). Schema (~280 tokens/table × 5 tables = ~1,400 tokens) is re-sent every loop iteration.

**Per question breakdown (medium complexity, 5 tables, 3 calls):**

| Component | Tokens |
|---|---|
| Schema (5 tables) | ~1,400 |
| System prompt | ~380 |
| ReAct template | ~210 |
| Question | ~35 |
| History re-sent | ~1,200 |
| DB results | ~440 |
| Output (all calls) | ~480 |
| **Total** | **~4,145 input + ~480 output** |

---

## 4. Baseline Code (30 lines)

```python
# pip install langchain langchain-anthropic langchain-community sqlalchemy

import os
from langchain_anthropic import ChatAnthropic
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# 1. Connect to SQLite (Kaggle file)
db = SQLDatabase.from_uri(
    "sqlite:///database.sqlite",
    include_tables=["Match", "Player", "Team", "League", "Player_Attributes"]
)

# 2. LLM — Claude
llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0,
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

# 3. Create agent
agent = create_sql_agent(
    llm=llm,
    db=db,
    verbose=True,           # shows SQL it generates
    handle_parsing_errors=True
)

# 4. Ask in plain English
result = agent.invoke("Which team scored the most goals across all seasons?")
print(result["output"])
```

**Swap to DeepSeek** (one line change):
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url="https://api.deepseek.com",
    model="deepseek-v4-flash",
    api_key=os.environ["DEEPSEEK_API_KEY"]
)
```

---

## 5. Three Agent Approaches

### Approach 1: ReAct Agent Loop (default)
- What: LangChain's `create_sql_agent` default. Thinks step-by-step in a loop: Thought → SQL → Observe → Answer.
- API calls: 2–4 per question
- Pro: Auto error recovery, familiar pattern
- Con: History re-sent every call, highest token cost
- Best for: Getting started, complex queries that might fail first try

### Approach 2: Single-Shot Prompt
- What: Skip the agent framework. One prompt with schema + question, LLM writes SQL, explains result.
- API calls: 1 per question
- Pro: Cheapest, most predictable output (good for screenshots)
- Con: No auto retry if SQL fails, you write the prompt
- Best for: Demos where you control the questions, social media recording

```python
prompt = f"""
Schema: {db.get_table_info()}
Question: {question}
Write SQL, show the query, explain the result.
"""
response = llm.invoke(prompt)
```

### Approach 3: Haiku + Sonnet Split
- What: Use Haiku (cheap) for SQL generation, Sonnet (capable) for final answer only.
- API calls: 2 per question
- Pro: ~40% cheaper than all-Sonnet, keeps quality where it matters
- Con: More code to manage, 2 API calls
- Best for: Production-ish builds, learning model routing patterns

---

## 6. Token Cost Optimisations

In order of impact:

| Optimisation | Savings | How |
|---|---|---|
| `include_tables` | ~45% | Expose only 5 relevant tables, not all 11 |
| Use Haiku for SQL gen | ~55% | Route SQL step to cheaper model |
| Single-shot (no agent loop) | ~38% | Eliminate loop overhead |
| Shorter system prompt | ~12% | Replace verbose ReAct template |
| Cache schema per session | ~18% | Read schema once, reuse for follow-ups |
| Read-only DB connection | Safety only | No cost saving but critical for safety |

**Rule of thumb:** `include_tables` is always the first thing to do. Non-negotiable.

---

## 7. API Costs & Trial Credits

### Current Pricing (May 2026, per million tokens)

| Model | Input | Output | Notes |
|---|---|---|---|
| DeepSeek V4 Flash | $0.14 | $0.28 | Cheapest serious model |
| DeepSeek V4 Pro | $0.43 | $0.87 | 75% promo until May 31 2026 |
| Claude Haiku 4.5 | $1.00 | $5.00 | ~7× more than Flash |
| Claude Sonnet 4.6 | $3.00 | $15.00 | ~53× more than Flash |

### Cost Per Question (football agent, medium complexity, 5 tables)

| Model | Cost/question | Questions for $5 |
|---|---|---|
| DeepSeek V4 Flash | ~$0.00083 | ~6,000 |
| DeepSeek V4 Pro | ~$0.0025 | ~2,000 |
| Claude Haiku 4.5 | ~$0.0055 | ~900 |
| Claude Sonnet 4.6 | ~$0.022 | ~225 |

### Free Trial Credits

| Provider | Free credit | Card required | Expiry |
|---|---|---|---|
| Anthropic | ~$5 | No (phone only) | None stated |
| DeepSeek | 5M tokens (~$8–10) | No | 30 days |

**Verdict:** Both trials are enough for a personal demo. DeepSeek's 5M token grant gives far more runway (~6,000–15,000 questions). Use DeepSeek for development/testing, Claude Sonnet for the final demo recording.

---

## 8. DeepSeek vs Claude — Key Comparison

### Performance by Task (approximate benchmark scores)

| Task | DeepSeek V4 Flash | DeepSeek V4 Pro | Claude Sonnet 4.6 |
|---|---|---|---|
| SQL / code generation | 72 | 78 | 66 |
| Agentic tasks | 62 | 70 | 65 |
| General reasoning | 47 | 55 | 47 |
| Knowledge & facts | 55 | 63 | 74 |
| Instruction following | 70 | 74 | 76 |

**For SQL generation specifically:** DeepSeek V4 Flash is more than capable — SQL is a structured, well-bounded task where it excels at its price point. Claude leads on knowledge and factual accuracy tasks.

### Caveats for DeepSeek

1. **Data privacy:** Servers in China. Prompts and DB schema route through Chinese infrastructure. Fine for a personal football demo. Serious issue for sensitive/business data.
2. **API availability:** Has frozen API top-ups during traffic spikes multiple times in 2025. Don't build production without a fallback.
3. **Drop-in compatible:** Uses same OpenAI API format — one line change in LangChain, zero other changes.

### Recommended Strategy for This Project
- Development/testing → DeepSeek V4 Flash (use the 5M free tokens)
- Final demo recording → Claude Sonnet 4.6 (polished output, recognisable brand)

---

## 9. Visualisation Layer

### What Changes Architecturally

The SQL agent step is **unchanged**. Two lightweight steps are added after results come back:

```
User question
      ↓
SQL Agent → Database → Raw results      ← unchanged
      ↓
Chart router (new, cheap)               ← classifies result type
      ↓
Parallel generation (new)
  ├── Branch A: Natural language answer  ← same as before
  └── Branch B: Chart spec JSON (new)   ← Vega-Lite or Plotly JSON
      ↓
Frontend renders both
```

### Three Visualisation Approaches

**Approach A — Tool use**
- LLM is given a `create_chart` tool; calls it when appropriate
- 1 extra API call, most elegant
- Extra tokens: ~350 input + ~280 output

**Approach B — Structured JSON output** ← recommended starting point
- One extra LLM call after SQL results: "return only a Vega-Lite spec"
- 1 extra API call, 20 extra lines of code, no new framework
- Extra tokens: ~620 input + ~380 output

**Approach C — Parallel calls**
- Fires answer + chart calls simultaneously with `asyncio`
- Same tokens as B, but faster wall-clock time
- Extra tokens: ~620 input + ~380 output

### Token Cost Overhead (Approach B, per question)

| Model | Base | With viz | Overhead |
|---|---|---|---|
| DeepSeek V4 Flash | $0.00083 | $0.00107 | +29% |
| Claude Haiku 4.5 | $0.0055 | $0.0074 | +35% |
| Claude Sonnet 4.6 | $0.022 | $0.030 | +36% |

**Takeaway:** The visualisation overhead is small in absolute terms. LLMs generate chart specs (structured JSON) efficiently — a Vega-Lite bar chart spec is ~300–400 output tokens.

### Recommended Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Chart spec format | Vega-Lite | LLMs trained on many examples, declarative JSON, renders with one line |
| Chart library | Plotly | Richer interactivity, works natively in Streamlit |
| Frontend/UI | Streamlit | 10 lines to a working demo UI, st.plotly_chart() built-in |
| Alternative UI | Gradio | Even simpler, auto-generates a shareable link |

### Minimal Working App (Streamlit + Vega-Lite + DeepSeek)

```python
# app.py — run with: streamlit run app.py
# pip install streamlit langchain langchain-openai langchain-community sqlalchemy altair

import streamlit as st
import json, os
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent

db = SQLDatabase.from_uri(
    "sqlite:///database.sqlite",
    include_tables=["Match", "Player", "Team", "League", "Player_Attributes"]
)

llm = ChatOpenAI(
    base_url="https://api.deepseek.com",
    model="deepseek-v4-flash",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    temperature=0
)

agent = create_sql_agent(llm=llm, db=db, handle_parsing_errors=True)

st.title("⚽ Football Stats Explorer")
question = st.text_input("Ask anything about the data")

if question:
    with st.spinner("Thinking..."):
        result = agent.invoke(question)
        st.write(result["output"])  # natural language answer

        chart_prompt = f"""The SQL result was: {result}
Question was: {question}
Return ONLY a valid Vega-Lite JSON spec with data inline.
If not visualisable, return: {{"$schema": "none"}}"""

        spec_raw = llm.invoke(chart_prompt).content
        try:
            spec = json.loads(spec_raw)
            if spec.get("$schema") != "none":
                st.vega_lite_chart(spec, use_container_width=True)
        except json.JSONDecodeError:
            pass  # silently skip if spec generation fails
```

---

## 10. Good Demo Queries for Social Media

| Category | Query | SQL complexity |
|---|---|---|
| Aggregation | "Which team scored the most goals across all seasons?" | SUM + GROUP BY + JOIN |
| Ranking | "Who are the top 5 players by overall FIFA rating?" | ORDER BY + LIMIT + JOIN |
| Cross-table | "Which league had the highest average goals per match?" | AVG + JOIN League + GROUP BY |
| Filtering | "How many matches ended in a draw in the 2015/2016 Premier League?" | WHERE + COUNT + season filter |
| Time comparison | "Which players improved their rating the most between 2010 and 2015?" | Self-join + date filter + delta |
| Insight | "Do taller players tend to have better overall ratings?" | AVG + CASE WHEN + correlation |

**Social media tip:** Screenshot the verbose=True output showing the SQL the agent generated alongside the plain English question and answer. That's the moment that makes people stop scrolling.

---

## 11. Next Steps (Build Order)

1. Download the Kaggle SQLite file
2. Set up API key (DeepSeek or Anthropic)
3. Run the 30-line baseline script in terminal — verify it works
4. Switch to single-shot approach for cost efficiency
5. Add Streamlit UI (text input + st.write for answer)
6. Add Vega-Lite chart generation (Approach B)
7. Polish UI for demo recording
8. Record and post — include a split showing question → SQL → chart

---

## 12. Open Questions / Future Directions

- Add prompt caching for schema (saves ~18% on repeated questions in same session)
- Try DeepSeek V4 Pro for more complex multi-table queries
- Explore adding a "follow-up question" feature (pass conversation history)
- Consider self-hosting DeepSeek if data privacy becomes a concern
- Evaluate adding a guardrail layer to detect and block destructive SQL

---

*Document auto-generated from project conversation. Update as the build evolves.*
