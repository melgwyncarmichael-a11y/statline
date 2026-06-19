import os, sys, sqlite3, random, time
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Statline · Hoops", page_icon="🏀", layout="wide")

# ── Shared utilities (parent directory) ───────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import (
    fresh_tokens, route_question, resolve_entities,
    single_shot, react, get_chart_spec, render_result,
    render_sidebar_history, cost_caption, compute_cost,
    ROUTE_LABELS, SUCCESS, OUT_OF_SCOPE,
)
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine

DB_PATH = os.path.join(os.path.dirname(__file__), "NBA Data", "nba.sqlite")

# ── Dataset config ────────────────────────────────────────────────────────────
DATASET_CONTEXT = (
    "NBA basketball statistics — 60,192 regular season games, 4,815 players, "
    "30 teams, seasons 1946 to 2023. "
    "Tables: game, player, team, common_player_info, line_score. "
    "Enrichment: player_salary (annual salaries 2000-2025) joined via player_name_lookup."
)

SCHEMA_HINTS = """
Key tables and what they contain:
- game: one row per game, has both home and away stats in the same row.
  Points: pts_home, pts_away. Win/loss: wl_home and wl_away ('W' or 'L').
  Team names are already in the table: team_name_home, team_name_away — no join needed for basic game queries.
  season_id format: '22021' = 2021-22 regular season (first digit is season type, 2 = regular season).
  Always filter regular season only with: season_id LIKE '2%'

- player: basic info only (id, full_name, is_active). For height, weight, position, draft info use common_player_info.
- common_player_info: detailed player profiles. Join: player.id = common_player_info.person_id.
- team: full team details. Join: game.team_id_home = team.id or game.team_id_away = team.id.
- line_score: quarter-by-quarter scores per game. Join: line_score.game_id = game.game_id.

To get total points scored by a team across games, SUM both pts_home (when home) and pts_away (when away) using UNION ALL.
To filter by season year: season_id LIKE '2YYYY%' (e.g. '22021%' for 2021-22).

Salary enrichment (Transfermarkt-style, 2000-2025, ~97% of players matched):
- player_salary(player_name, player_name_normalised, salary, season_start_year): one row per player per season.
  salary is in US dollars; season_start_year is an integer (2021 = the 2021-22 season).
- player_name_lookup(player_id, full_name, full_name_normalised): bridge table mapping salary names to player.id.
- To join salaries to a player:
    player_salary ps
    JOIN player_name_lookup pnl ON pnl.full_name_normalised = ps.player_name_normalised
    JOIN player p ON p.id = pnl.player_id
  Always join through player_name_lookup — never match player_salary.player_name to player.full_name directly
  (spelling/accent differences mean the direct match misses ~3% of rows).
- To align a salary season with the game table: ps.season_start_year = CAST(SUBSTR(g.season_id, 2, 4) AS INT).
- LIMITATION: player_salary has NO team column, and common_player_info.team_name/team_id is only the player's
  CURRENT team, not their team in a past season. Team-level payroll for a given season therefore CANNOT be
  computed. If asked for a team's payroll/total salary in a season, do NOT join game or common_player_info to
  fabricate a team attribution (that also fans out and inflates the total). Instead return exactly:
    SELECT 'Team payroll by season cannot be computed — salary data has no team, and only current team is known.' AS answer;
- player_salary is per-player, NEVER join it to the game table (game has no salary and causes a huge fan-out).
"""

ENTITY_TABLES = [
    ("team",               "full_name"),
    ("player",             "full_name"),
    ("common_player_info", "display_first_last"),
]

QUESTION_CATEGORIES = {
    "🏆 Rankings": [
        "Who are the top 10 scorers by total points all time?",
        "Which team has the most wins in NBA history?",
    ],
    "📊 Season Stats": [
        "Which team scored the most points in the 2021-22 season?",
        "Which team had the best win rate in the 2020-21 regular season?",
    ],
    "👤 Player Stats": [
        "What is the average height of NBA players by position?",
        "How many players in the dataset were drafted from international schools?",
    ],
    "📈 Trends": [
        "What is the average points per game by season?",
        "How many games went to overtime?",
    ],
    "💰 Salaries": [
        "Who were the 10 highest-paid players in the 2021 season?",
        "How has the average player salary changed over the years?",
    ],
}
QUESTIONS = [q for qs in QUESTION_CATEGORIES.values() for q in qs]

OUT_OF_SCOPE_MSG = (
    "⚠️ **This question is outside the dataset.**\n\n"
    "This app only knows about NBA data: games, teams, players and stats "
    "from **1946 to 2023**.\n\n"
    "Try asking:\n"
    "- *Which team scored the most points in the 2021-22 season?*\n"
    "- *Who are the top 10 scorers of all time?*\n"
    "- *Which team had the best win rate in the 2019-20 season?*"
)


# ── Overview data ─────────────────────────────────────────────────────────────

@st.cache_data
def get_overview():
    conn = sqlite3.connect(DB_PATH)
    wins_by_team = pd.read_sql("""
        SELECT team_name_home AS Team, COUNT(*) AS Wins
        FROM game WHERE wl_home = 'W' AND season_id LIKE '2%'
        GROUP BY team_name_home
        UNION ALL
        SELECT team_name_away AS Team, COUNT(*) AS Wins
        FROM game WHERE wl_away = 'W' AND season_id LIKE '2%'
        GROUP BY team_name_away
    """, conn)
    wins_by_team = (
        wins_by_team.groupby("Team", as_index=False)["Wins"].sum()
        .sort_values("Wins", ascending=False).head(15)
    )
    avg_pts_by_season = pd.read_sql("""
        SELECT SUBSTR(season_id, 2, 4) AS Season,
               ROUND(AVG(pts_home + pts_away), 1) AS "Avg Total Points"
        FROM game WHERE season_id LIKE '2%' AND pts_home IS NOT NULL
        GROUP BY SUBSTR(season_id, 2, 4) ORDER BY Season
    """, conn)
    top_players = pd.read_sql("""
        SELECT display_first_last AS Player, team_name AS Team,
               position AS Position,
               CAST(from_year AS INT) AS "From",
               CAST(to_year   AS INT) AS "To"
        FROM common_player_info
        WHERE rosterstatus = 'Active'
        ORDER BY season_exp DESC LIMIT 10
    """, conn)
    conn.close()
    return wins_by_team, avg_pts_by_season, top_players


@st.cache_data
def get_schema_explorer():
    conn = sqlite3.connect(DB_PATH)
    tables = ["game", "player", "team", "common_player_info", "line_score",
              "player_salary", "player_name_lookup"]
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
        include_tables=["game", "player", "team", "common_player_info", "line_score",
                        "player_salary", "player_name_lookup"]
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

st.title("🏀 Statline · Hoops")
st.caption("Ask anything about the NBA — in plain English.")

st.markdown("""
**Dataset:** NBA Database (Kaggle) — comprehensive NBA data spanning 1946 to 2023.
Contains 60,192 regular season games, 4,815 players, and 30 teams, including game-by-game stats,
quarter-by-quarter scores, player profiles, and draft history.
""")

c1, c2, c3, c4 = st.columns(4)
c1.metric("🏀 Games",   "60,192")
c2.metric("👤 Players", "4,815")
c3.metric("🏆 Teams",   "30")
c4.metric("📅 Seasons", "1946 – 2023")

with st.expander("📊 Data Overview — click to explore the dataset"):
    wins_by_team, avg_pts_by_season, top_players = get_overview()
    col1, col2 = st.columns(2)
    with col1:
        st.caption("🏆 All-time wins by team (regular season, top 15)")
        st.bar_chart(wins_by_team.set_index("Team"), height=260)
    with col2:
        st.caption("📈 Average total points per game by season")
        st.line_chart(avg_pts_by_season.set_index("Season"), height=260)
    st.caption("🌟 Most experienced active players")
    st.dataframe(top_players, use_container_width=True, hide_index=True)

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
        placeholder="e.g. Which team scored the most points in the 2021-22 season?"
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
