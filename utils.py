"""
Shared logic for Football SQL Agent and NBA Stats Explorer.
Dataset-specific config, overview queries, resource loading, and UI stay in each app's app.py.
"""
import re, json, time
import pandas as pd
import streamlit as st
from langchain_community.callbacks import get_openai_callback
from sqlalchemy import text

# ── Pricing (DeepSeek V4 Flash, May 2026) ────────────────────────────────────
INPUT_PRICE_PER_M  = 0.14
OUTPUT_PRICE_PER_M = 0.28
MODEL_LABEL        = "DeepSeek V4 Flash"

# ── State constants ───────────────────────────────────────────────────────────
SUCCESS, OUT_OF_SCOPE, TOO_COMPLEX = "success", "out_of_scope", "too_complex"
SQL_ERROR, EMPTY_RESULT, PARSE_ERROR = "sql_error", "empty", "parse_error"

# ── Routing constants ─────────────────────────────────────────────────────────
REACT_TRIGGERS = [
    r'\bvs\b', r'\bversus\b', r'\bhead.to.head\b', r'\bcompared? to\b',
    r'\bimproved?\b', r'\bchange[d]?\b', r'\bbetween\b.*\band\b',
    r'\bfrom\b.*\bto\b', r'\bover time\b', r'\byear.over.year\b', r'\bmost improved\b',
]

ROUTE_LABELS = {
    "SIMPLE":       "⚡ Single-shot",
    "ENTITY":       "🔍 Entity lookup → single-shot",
    "COMPLEX":      "🔄 ReAct loop",
    "OUT_OF_SCOPE": "🚫 Out of scope",
}


# ── Token tracking ────────────────────────────────────────────────────────────

def fresh_tokens():
    return {"input": 0, "output": 0}


def invoke_tracked(llm, prompt: str, tokens: dict) -> str:
    response = llm.invoke(prompt)
    try:
        usage = response.response_metadata.get("token_usage", {})
        tokens["input"]  += usage.get("prompt_tokens", 0)
        tokens["output"] += usage.get("completion_tokens", 0)
    except Exception:
        pass
    return response.content


def compute_cost(tokens: dict) -> float:
    """Single source of truth for token → dollar cost. Both apps and the badges use this."""
    return (tokens["input"] / 1e6 * INPUT_PRICE_PER_M +
            tokens["output"] / 1e6 * OUTPUT_PRICE_PER_M)


def cost_caption(tokens: dict) -> str:
    total = tokens["input"] + tokens["output"]
    return (f"💰 **{total:,} tokens** ({tokens['input']:,} in · {tokens['output']:,} out) · "
            f"**~${compute_cost(tokens):.6f}** · {MODEL_LABEL}")


# ── Router ────────────────────────────────────────────────────────────────────

def route_question(question: str, llm, tokens: dict, dataset_context: str) -> str:
    """Classify the question into SIMPLE / ENTITY / COMPLEX / OUT_OF_SCOPE."""
    if any(re.search(p, question.lower()) for p in REACT_TRIGGERS):
        return "COMPLEX"
    prompt = f"""You are routing queries for a database about: {dataset_context}

Classify the question with exactly one word:
- OUT_OF_SCOPE: has nothing to do with the dataset above
- SIMPLE: straightforward aggregation or ranking, no specific named entities
- ENTITY: mentions specific team or player names that need verifying in the database
- COMPLEX: requires comparing across time periods, self-joins, or multi-step reasoning

Question: {question}
Answer (one word only):"""
    result = invoke_tracked(llm, prompt, tokens).strip().upper()
    word = result.split()[0] if result.split() else "SIMPLE"
    return word if word in ("OUT_OF_SCOPE", "SIMPLE", "ENTITY", "COMPLEX") else "SIMPLE"


def resolve_entities(question: str, engine, entity_tables: list) -> str:
    """Look up capitalised words in the DB to find exact name matches."""
    candidates = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', question)
    hints, seen = [], set()
    with engine.connect() as conn:
        for candidate in candidates:
            if candidate in seen or len(candidate) < 3:
                continue
            seen.add(candidate)
            for table, column in entity_tables:
                try:
                    rows = conn.execute(
                        text(f'SELECT DISTINCT {column} FROM "{table}" WHERE {column} LIKE :p LIMIT 3'),
                        {"p": f"%{candidate}%"}
                    ).fetchall()
                    if rows:
                        hints.append(f"'{candidate}' found in DB as: {[r[0] for r in rows]}")
                except Exception:
                    pass
    return "\n".join(hints)


# ── Query execution ───────────────────────────────────────────────────────────

def run_query(engine, sql: str) -> pd.DataFrame:
    """Run a raw SQL string and return a DataFrame."""
    with engine.connect() as conn:
        return pd.read_sql_query(text(sql), conn)


def single_shot(question: str, engine, llm, schema: str, tokens: dict,
                schema_hints: str, extra_hints: str = ""):
    """Generate SQL in one LLM call, execute it, and return (state, sql, df, explanation)."""
    hint_block = f"\nEntity names resolved from DB:\n{extra_hints}" if extra_hints else ""
    prompt = f"""You are a SQL expert. Given the schema and hints below, write a SQLite query to answer the question.

Schema:
{schema}

Hints:
{schema_hints}{hint_block}

Question: {question}

Reply in this exact format:
SQL:
```sql
<your query here>
```
Answer: <one sentence explaining what the result shows>"""

    response = invoke_tracked(llm, prompt, tokens)
    sql_match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL)
    if not sql_match:
        return PARSE_ERROR, None, None, "Could not extract SQL from the response."
    sql = sql_match.group(1).strip()
    ans_match = re.search(r"Answer:\s*(.+)", response)
    explanation = ans_match.group(1).strip() if ans_match else ""
    try:
        df = run_query(engine, sql)
    except Exception as e:
        return SQL_ERROR, sql, None, str(e)
    return (EMPTY_RESULT if df.empty else SUCCESS), sql, df, explanation


def react(question: str, agent, engine, tokens: dict):
    """Run the LangChain ReAct SQL agent and return (state, sql, df, answer)."""
    try:
        with get_openai_callback() as cb:
            result = agent.invoke(question)
        tokens["input"]  += cb.prompt_tokens
        tokens["output"] += cb.completion_tokens
    except Exception as e:
        return TOO_COMPLEX, None, None, str(e)

    answer = result.get("output", "")
    if "iteration limit" in answer.lower() or "time limit" in answer.lower() or not answer:
        return TOO_COMPLEX, None, None, answer

    sql, df = None, None
    for action, _ in result.get("intermediate_steps", []):
        if hasattr(action, "tool") and action.tool == "sql_db_query":
            sql = action.tool_input
    if sql:
        try:
            df = run_query(engine, sql)
        except Exception:
            df = None
    return SUCCESS, sql, df, answer


# ── Chart generation ──────────────────────────────────────────────────────────

def get_chart_spec(question: str, df, llm, tokens: dict):
    """Ask the LLM for a Vega-Lite spec; returns None if the data isn't chartable."""
    if df is None or df.empty or len(df.columns) < 2:
        return None
    sample = df.head(20).to_dict(orient="records")
    prompt = f"""You are a data visualisation expert. Return a Vega-Lite JSON spec with data inline.

Question: {question}
Data (up to 20 rows): {json.dumps(sample)}

Rules:
- Use mark: bar, line, or point only
- x axis: categorical or date field, y axis: numeric field
- Include a descriptive title
- If not visualisable, return exactly: {{"$schema": "none"}}
- Return ONLY valid JSON, no explanation"""
    raw = invoke_tracked(llm, prompt, tokens).strip()
    raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        spec = json.loads(raw)
        return None if spec.get("$schema") == "none" else spec
    except json.JSONDecodeError:
        return None


# ── Result renderer ───────────────────────────────────────────────────────────

def render_result(state, sql, df, answer, chart_spec, tokens, out_of_scope_msg: str,
                  route: str = None, elapsed: float = None):
    """Render the final answer, table, chart, and SQL expander in Streamlit."""
    if state == OUT_OF_SCOPE:
        st.warning(out_of_scope_msg)
        return

    if state == TOO_COMPLEX:
        st.error(
            "🤯 **This question was too complex to answer reliably.**\n\n"
            "The agent ran out of steps before reaching a confident answer. "
            "Try breaking it into simpler parts — ask about one team or one season at a time."
        )
        return

    if state == SQL_ERROR:
        st.error(
            f"❌ **SQL error — the generated query failed to run.**\n\n"
            f"The question might reference a name that doesn't exactly match the database.\n\n"
            f"**Error:** `{answer}`"
        )
        if sql:
            with st.expander("Failed SQL query"):
                st.code(sql, language="sql")
        return

    if state == PARSE_ERROR:
        st.error("❌ **Could not generate a valid SQL query.**\n\nTry rephrasing your question.")
        return

    if state == EMPTY_RESULT:
        st.info(
            "ℹ️ **No matching data found.**\n\n"
            "The query ran but returned zero rows. The condition might be too strict, "
            "or that scenario didn't occur in the dataset."
        )
        if sql:
            with st.expander("SQL query"):
                st.code(sql, language="sql")
        return

    # ── SUCCESS ──

    # Route / latency / cost badge
    total_cost = compute_cost(tokens)
    total_tokens = tokens["input"] + tokens["output"]
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Route", ROUTE_LABELS.get(route, "—") if route else "—")
    b2.metric("Latency", f"{elapsed:.1f}s" if elapsed is not None else "—")
    b3.metric("Tokens", f"{total_tokens:,}")
    b4.metric("Cost", f"${total_cost:.5f}")

    st.markdown("### Answer")
    st.write(answer)

    if df is not None and not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            label="⬇️ Download as CSV",
            data=df.to_csv(index=False),
            file_name="results.csv",
            mime="text/csv"
        )

    if chart_spec:
        st.vega_lite_chart(chart_spec, use_container_width=True)

    if sql:
        with st.expander("SQL query"):
            st.code(sql, language="sql")


# ── Sidebar history ───────────────────────────────────────────────────────────

def render_sidebar_history(history: list):
    """Render the last 3 successful queries in the sidebar."""
    with st.sidebar:
        st.markdown("### 🕐 Recent Queries")
        if not history:
            st.caption("No queries yet — ask something above.")
            return
        for entry in history:
            label = entry["question"]
            short = label[:48] + "…" if len(label) > 48 else label
            with st.expander(short):
                st.markdown(f"**{entry['route_label']}** · {entry['elapsed']:.1f}s · ${entry['cost']:.5f}")
                answer_text = entry["answer"]
                st.write(answer_text[:300] + ("…" if len(answer_text) > 300 else ""))
