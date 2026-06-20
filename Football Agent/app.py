import os, sys, sqlite3, random, time
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Statline · Football", page_icon="⚽", layout="wide")

# ── Shared utilities (parent directory) ───────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import (
    fresh_tokens, route_question, resolve_entities,
    single_shot, react, get_chart_spec, render_result,
    render_sidebar_history, cost_caption, compute_cost,
    inject_theme, render_hero,
    ROUTE_LABELS, SUCCESS, OUT_OF_SCOPE,
)

# ── Brand accent (pitch green) ────────────────────────────────────────────────
ACCENT      = "#1B9E55"
ACCENT_SOFT = "#E8F6EE"
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine

DB_PATH = os.path.join(os.path.dirname(__file__), "database.sqlite")

# ── Dataset config ────────────────────────────────────────────────────────────
DATASET_CONTEXT = (
    "European football statistics — 11 leagues (Premier League, La Liga, Bundesliga, "
    "Serie A, etc.), 25,979 matches, 11,060 players, seasons 2008 to 2016. "
    "Tables: Match, Team, Player, Player_Attributes, League. "
    "Enrichment (Transfermarkt, 2008-2016): player_valuations (historical market value in EUR) "
    "and player_appearances (per-match goals, assists, cards, minutes). "
    "The Match table also holds bookmaker betting odds (Bet365 etc.) for ~87% of matches."
)

SCHEMA_HINTS = """
Key join: Player.player_api_id = Player_Attributes.player_api_id
Key join: Match.home_team_api_id = Team.team_api_id (and away_team_api_id)
Key join: Match.league_id = League.id
Goals scored by a team = home_team_goal (when home) + away_team_goal (when away) — use UNION ALL, not OR join.
Player ratings are in Player_Attributes (overall_rating, potential, finishing, etc).

Enrichment tables (Transfermarkt data, ~67% of players, 2008-2016 only):
- player_valuations(player_api_id, date, market_value_in_eur, current_club_name): one row per valuation snapshot.
  Join to players via player_valuations.player_api_id = Player.player_api_id.
  date is a string 'YYYY-MM-DD'; filter a year with date LIKE '2013%'. A player has MANY rows over time —
  use MAX(market_value_in_eur) for peak value, or pick a specific date window. Values are in euros.
- player_appearances(player_api_id, game_id, date, competition_id, goals, assists, yellow_cards, red_cards, minutes_played):
  one row per player per match. Join via player_appearances.player_api_id = Player.player_api_id.
  SUM(goals)/SUM(assists) for season or career totals; filter date with BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'.
  NOTE: these are REAL match goals/assists, distinct from FIFA potential ratings in Player_Attributes.
Only ~67% of players have enrichment rows; a LEFT JOIN avoids dropping unmatched players, but for
"top by value/goals" questions an inner JOIN is fine since unmatched players have no data anyway.

CRITICAL — never JOIN player_valuations and player_appearances directly to each other: both have many
rows per player, so the join fans out (multiplies) and inflates SUM(goals)/SUM(value). When a question
needs BOTH (e.g. "players worth over 50M who scored the most goals"), filter one table with a subquery:
  SELECT p.player_name, SUM(pa.goals) AS goals
  FROM player_appearances pa JOIN Player p ON p.player_api_id = pa.player_api_id
  WHERE pa.player_api_id IN (
      SELECT player_api_id FROM player_valuations WHERE market_value_in_eur > 50000000)
  GROUP BY p.player_api_id ORDER BY goals DESC;

Betting odds (in Match table, decimal odds, ~87% coverage for Bet365):
- Each bookmaker has three columns suffixed H/D/A = Home win / Draw / Away win.
  Bet365 = B365H/B365D/B365A (best coverage; prefer it). Others: BW, IW, LB, PS, WH, SJ, VC, GB, BS.
- LOWER odds = more likely outcome (the favourite). The favourite is the side with the smallest
  of B365H/B365D/B365A. Implied probability = 1 / odds.
- A team was the "underdog" in a match when its win odds were HIGHER than the opponent's
  (home underdog: B365H > B365A; away underdog: B365A > B365H).
- Always filter "WHERE B365H IS NOT NULL" for odds questions to skip matches without odds.

CRITICAL — for "which club / squad value" questions, use player_valuations.current_club_name DIRECTLY
(it records the player's club at the time of each valuation snapshot). NEVER join player_valuations or
player_appearances to the Match table to figure out a player's club — Match.home_player_1..11 are per-match
lineup IDs and such a join fans out massively (producing values in the billions). For a club's total squad
value in a year, take the latest snapshot per player that year, then SUM by current_club_name:
  WITH latest AS (
    SELECT player_api_id, current_club_name, market_value_in_eur,
           ROW_NUMBER() OVER (PARTITION BY player_api_id ORDER BY date DESC) rn
    FROM player_valuations WHERE date LIKE '2013%')
  SELECT current_club_name, SUM(market_value_in_eur) FROM latest WHERE rn = 1 GROUP BY current_club_name;
"""

ENTITY_TABLES = [
    ("Team",   "team_long_name"),
    ("Player", "player_name"),
    ("League", "name"),
]

QUESTION_CATEGORIES = {
    "🏆 Rankings": [
        "Which team scored the most goals across all seasons?",
        "Who are the top 5 players by overall FIFA rating?",
    ],
    "📊 League Stats": [
        "Which league had the highest average goals per match?",
        "How many matches ended in a draw in the 2015/2016 Premier League?",
    ],
    "⚔️ Comparisons": [
        "What was FC Barcelona's win rate at home vs away?",
        "Which players improved their rating the most between 2010 and 2015?",
    ],
    "📈 Trends": [
        "Do taller players tend to have better overall ratings?",
        "Which season had the most total goals scored?",
    ],
    "💰 Market Value": [
        "Who were the 10 most valuable players in 2013?",
        "Which players scored the most real goals between 2010 and 2014?",
    ],
    "🎲 Betting Odds": [
        "How often did the Bet365 favorite actually win the match?",
        "Which team won the most matches as the underdog?",
    ],
}
QUESTIONS = [q for qs in QUESTION_CATEGORIES.values() for q in qs]

OUT_OF_SCOPE_MSG = (
    "⚠️ **This question is outside the dataset.**\n\n"
    "This app only knows about European football: matches, teams, players, "
    "leagues, FIFA ratings, market values and match stats from **2008 to 2016**.\n\n"
    "Try asking:\n"
    "- *Which team scored the most goals in La Liga?*\n"
    "- *Who had the highest overall rating in 2014?*\n"
    "- *Who were the most valuable players in 2013?*"
)


# ── Overview data ─────────────────────────────────────────────────────────────

@st.cache_data
def get_overview():
    conn = sqlite3.connect(DB_PATH)
    goals_by_league = pd.read_sql("""
        SELECT l.name AS League, SUM(m.home_team_goal + m.away_team_goal) AS Goals
        FROM Match m JOIN League l ON m.league_id = l.id
        GROUP BY l.name ORDER BY Goals DESC
    """, conn)
    matches_by_season = pd.read_sql("""
        SELECT season AS Season, COUNT(*) AS Matches
        FROM Match GROUP BY season ORDER BY season
    """, conn)
    top_players = pd.read_sql("""
        SELECT p.player_name AS Player,
               MAX(pa.overall_rating) AS "Overall Rating",
               MAX(pa.potential)      AS Potential
        FROM Player_Attributes pa JOIN Player p ON pa.player_api_id = p.player_api_id
        GROUP BY p.player_name ORDER BY "Overall Rating" DESC LIMIT 10
    """, conn)
    top_valued = pd.read_sql("""
        SELECT p.player_name AS Player,
               MAX(pv.market_value_in_eur) / 1e6 AS "Peak Value (€M)"
        FROM player_valuations pv JOIN Player p ON pv.player_api_id = p.player_api_id
        GROUP BY p.player_api_id ORDER BY "Peak Value (€M)" DESC LIMIT 10
    """, conn)
    conn.close()
    return goals_by_league, matches_by_season, top_players, top_valued


@st.cache_data
def get_schema_explorer():
    conn = sqlite3.connect(DB_PATH)
    tables = ["Match", "Team", "Player", "Player_Attributes", "League",
              "player_valuations", "player_appearances"]
    result = {}
    for table in tables:
        cols_df = pd.read_sql(f"PRAGMA table_info('{table}')", conn)
        try:
            sample = pd.read_sql(f'SELECT * FROM "{table}" LIMIT 1', conn)
        except Exception:
            sample = pd.DataFrame()
        rows = []
        for _, col in cols_df.iterrows():
            col_name = col["name"]
            example = "—"
            if not sample.empty and col_name in sample.columns:
                val = sample[col_name].iloc[0]
                example = "—" if pd.isna(val) else str(val)[:40]
            rows.append({"Column": col_name, "Type": col["type"] or "TEXT", "Example": example})
        result[table] = pd.DataFrame(rows)
    conn.close()
    return result


# ── Resource loading ──────────────────────────────────────────────────────────

@st.cache_resource
def load_resources():
    # Open the DB read-only so LLM-generated SQL can never mutate/DROP the demo data.
    # Covers every execution path: single_shot, entity lookups, and the ReAct agent's own engine.
    ro_uri = f"sqlite:///file:{DB_PATH}?mode=ro&uri=true"
    engine = create_engine(ro_uri)
    db = SQLDatabase.from_uri(
        ro_uri,
        include_tables=["Match", "Player", "Team", "League", "Player_Attributes",
                        "player_valuations", "player_appearances"]
    )
    llm = ChatOpenAI(
        base_url="https://api.deepseek.com", model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"], temperature=0
    )
    agent = create_sql_agent(
        llm=llm, db=db, handle_parsing_errors=True,
        max_iterations=12, verbose=False, return_intermediate_steps=True
    )
    schema = db.get_table_info()
    return engine, llm, agent, schema


# ── Session state ─────────────────────────────────────────────────────────────

if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "history" not in st.session_state:
    st.session_state.history = []

render_sidebar_history(st.session_state.history)

# ── UI ────────────────────────────────────────────────────────────────────────

inject_theme(ACCENT, ACCENT_SOFT)
render_hero("⚽", "Football", "Ask anything about European football — in plain English.",
            ACCENT, ACCENT_SOFT)

st.markdown("""
**Dataset:** European Soccer Database (Kaggle) — 11 European leagues including the Premier League,
La Liga, Bundesliga, and Serie A. Contains 25,979 matches, 11,060 players, and detailed FIFA
player attributes across 8 seasons (2008–2016), enriched with **Transfermarkt market values** and
**real per-match goals/assists** for ~67% of players.
""")

c1, c2, c3, c4 = st.columns(4)
c1.metric("🏆 Leagues", "11")
c2.metric("⚽ Matches", "25,979")
c3.metric("👤 Players", "11,060")
c4.metric("📅 Seasons", "2008 – 2016")

with st.expander("📊 Data Overview — click to explore the dataset"):
    goals_by_league, matches_by_season, top_players, top_valued = get_overview()
    col1, col2 = st.columns(2)
    with col1:
        st.caption("⚽ Total goals scored by league")
        st.bar_chart(goals_by_league.set_index("League"), height=260)
    with col2:
        st.caption("📅 Matches played per season")
        st.bar_chart(matches_by_season.set_index("Season"), height=260)
    col3, col4 = st.columns(2)
    with col3:
        st.caption("🌟 Top 10 players by overall FIFA rating")
        st.dataframe(top_players, use_container_width=True, hide_index=True)
    with col4:
        st.caption("💰 Top 10 players by peak market value")
        st.dataframe(top_valued, use_container_width=True, hide_index=True)

with st.expander("🗂️ Schema Explorer — browse tables and columns"):
    schema_data = get_schema_explorer()
    tabs = st.tabs(list(schema_data.keys()))
    for tab, (table_name, df) in zip(tabs, schema_data.items()):
        with tab:
            st.dataframe(df, use_container_width=True, hide_index=True, height=280)

with st.expander("💡 Questions you can ask — click one to try it"):
    for category, questions in QUESTION_CATEGORIES.items():
        st.markdown(f"**{category}**")
        cols = st.columns(2)
        for i, q in enumerate(questions):
            if cols[i % 2].button(q, key=f"chip_{category}_{i}", use_container_width=True):
                st.session_state.question_input = q
                st.rerun()

# ── Question input ────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### 💬 Ask a question about the data")
col_input, col_rand = st.columns([8, 1])
with col_rand:
    if st.button("🎲", help="Pick a random question", use_container_width=True):
        st.session_state.question_input = random.choice(QUESTIONS)
        st.rerun()
with col_input:
    question = st.text_input(
        "question",
        label_visibility="collapsed",
        key="question_input",
        placeholder="e.g. Which team scored the most goals across all seasons?"
    )

# ── Query handling ────────────────────────────────────────────────────────────

if question:
    tokens = fresh_tokens()
    t_start = time.time()

    with st.status("Working...", expanded=True) as status:

        t_step = time.time()
        engine, llm, agent, schema = load_resources()
        st.write(f"✅ Resources ready · {time.time() - t_step:.1f}s")

        t_step = time.time()
        route = route_question(question, llm, tokens, DATASET_CONTEXT)
        st.write(f"✅ Routed → **{ROUTE_LABELS[route]}** · {time.time() - t_step:.1f}s")

        chart_spec = None

        if route == "OUT_OF_SCOPE":
            state, sql, df, answer = OUT_OF_SCOPE, None, None, None

        elif route == "SIMPLE":
            t_step = time.time()
            state, sql, df, answer = single_shot(question, engine, llm, schema, tokens, SCHEMA_HINTS)
            st.write(f"✅ SQL generated & executed · {time.time() - t_step:.1f}s")

        elif route == "ENTITY":
            t_step = time.time()
            hints = resolve_entities(question, engine, ENTITY_TABLES)
            st.write(f"✅ Entities resolved · {time.time() - t_step:.1f}s")
            t_step = time.time()
            state, sql, df, answer = single_shot(
                question, engine, llm, schema, tokens, SCHEMA_HINTS, extra_hints=hints
            )
            st.write(f"✅ SQL generated & executed · {time.time() - t_step:.1f}s")

        else:
            t_step = time.time()
            state, sql, df, answer = react(question, agent, engine, tokens)
            st.write(f"✅ ReAct agent done · {time.time() - t_step:.1f}s")

        if state == SUCCESS:
            t_step = time.time()
            chart_spec = get_chart_spec(question, df, llm, tokens)
            st.write(f"✅ Chart built · {time.time() - t_step:.1f}s")

        elapsed = time.time() - t_start
        status.update(
            label=f"Done · {ROUTE_LABELS[route]} · {elapsed:.1f}s total",
            state="complete", expanded=False
        )

    render_result(state, sql, df, answer, chart_spec, tokens, OUT_OF_SCOPE_MSG,
                  route=route, elapsed=elapsed)

    if state == SUCCESS and answer:
        total_cost = compute_cost(tokens)
        st.session_state.history.insert(0, {
            "question": question,
            "route_label": ROUTE_LABELS[route],
            "answer": answer,
            "elapsed": elapsed,
            "cost": total_cost,
        })
        st.session_state.history = st.session_state.history[:3]
