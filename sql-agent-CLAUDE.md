# SQL Agent — Claude Instructions

## What This Project Is

A sports analytics demo with two parallel Streamlit apps — Football Agent and NBA Agent.
Both apps follow identical structure and must always stay in sync on features.
Built for demo recording and LinkedIn content. The "magic moment" is plain English
question → SQL → chart, side by side.

---

## Critical Rules

- **Never merge the two apps into one.** They stay separate intentionally.
- **Any feature change must be applied to both** `Football Agent/app.py` and `Nba Agent 2/app.py`.
  They differ only in: DB path, table names, SCHEMA_HINTS, ENTITY_TABLES,
  DATASET_CONTEXT, metrics row values, overview charts, and sample questions.
- **Never hardcode DEEPSEEK_API_KEY.** It is set in `launch.command` as an
  environment variable and must stay there.
- **Do not remove** `baseline.py` or `single_shot.py` — kept intentionally as
  reference implementations.
- **Do not add new dependencies** without checking if something built-in works first.
- **Do not change routing logic** without considering token cost impact.

---

## Architecture

### Routing
Single-shot by default. ReAct only when routed there.
ReAct costs 4x more tokens and re-sends the full schema every loop iteration.
Most demo questions don't need self-correction — the router decides automatically.

### Models
- Development: DeepSeek Flash (~21x cheaper than Claude Sonnet, benchmarks higher on SQL)
- Final recording: Claude Sonnet 4.6 (better instruction following, brand recognition)
- When switching to Claude: consider adding explicit `cache_control` markers
  (Anthropic supports them; DeepSeek does it automatically)

### Database
SQLite only. No Postgres, no MySQL. No server, no credentials, one file.

### Charts
Vega-Lite via `st.vega_lite_chart()`. Not Plotly. Built into Streamlit,
LLMs are well-trained on Vega-Lite JSON, no extra install needed.

### Schema
`include_tables` restricts schema to 5 relevant tables per dataset.
Reduces token cost ~45% per query. Non-negotiable optimisation.

### ReAct Agent
`max_iterations=12`. Do not lower this.
6 was too low — complex multi-table queries (3+ joins, self-joins, CTEs) need room.
`return_intermediate_steps=True` — extracts last SQL the agent ran and re-executes
it as a DataFrame so the chart generator has structured data on the ReAct path.

### Entity Resolution
Via DB lookup (SQL LIKE query), not a second LLM call. Free and fixes the main
class of schema trap failures (wrong team/player names).

---

## Common Mistakes — Never Repeat These

- **OR joins instead of UNION ALL** for goals/points queries. Always use UNION ALL
  to separate home and away contributions, then aggregate. Encoded in SCHEMA_HINTS.
- **Wrong team/player names.** DB has "FC Barcelona" and "Real Madrid CF" — not
  "Barcelona" and "Real Madrid". Entity resolution handles this automatically.
- **Including All-Star and playoff data** when user asks about "a season". Always
  filter `season_id LIKE '2%'` for regular season. In SCHEMA_HINTS.
- **max_iterations too low** causing "Agent stopped due to iteration limit" errors.
  It is 12. Do not lower it.
- **Chart generator returning JSON in markdown code fences.** The regex strip
  handles this — it must stay in place.

---

## Key Files

- `Football Agent/app.py` — live Football app
- `Nba Agent 2/app.py` — live NBA app (must mirror Football app on all features)
- `launch.command` — starts both apps, kills port conflicts, opens browser.
  Must stay in the SQL Agent root.

---

## Code Style

- Functions over classes
- Group related logic with `# ── Section name ──` comment banners
- Keep `SCHEMA_HINTS`, `REACT_TRIGGERS`, `ENTITY_TABLES` as top-level constants
- `@st.cache_resource` for LLM/DB/agent objects (per-session singletons)
- `@st.cache_data` for query results (serialisable, re-runnable)
- Explicit result state constants (`SUCCESS`, `TOO_COMPLEX`, etc.) — never check
  string content of error messages
- Functions: `snake_case`, verb-first (`run_query`, `resolve_entities`)
- Constants: `UPPER_SNAKE_CASE`
- Files: lowercase with underscores for Python, Title Case for folders

---

## How to Work on This Project

- Explain reasoning before making changes, especially architectural ones
- Make targeted edits — don't rewrite whole files unless multiple sections are changing
- Check what already exists before creating new files
- Always apply changes to both apps

---

## Current Status

Feature-complete. Finalising for demo recording.

**Coming next:**
- Switch LLM from DeepSeek to Claude Sonnet 4.6 for final recording
- Record the demo
- Post to LinkedIn as a series (one post per major build step)

**Open questions:**
- Whether to add explicit prompt caching when switching to Claude
- Whether ReAct path needs a token usage logger for the cost transparency post
