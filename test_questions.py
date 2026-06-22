#!/usr/bin/env python3
"""
Headless regression test — runs every chip question for both apps through the
real pipeline (route → SQL → execute) and reports pass/fail.
Run from the SQL Agent root: python test_questions.py
"""
import sys, os, time, textwrap
from unittest.mock import MagicMock

# Streamlit is UI-only; mock it so utils imports cleanly in a terminal context.
sys.modules["streamlit"] = MagicMock()

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# Load .env
env_path = os.path.join(BASE, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

from utils import (
    route_question, resolve_entities, single_shot, react,
    fresh_tokens, SUCCESS, SQL_ERROR, EMPTY_RESULT, TOO_COMPLEX, PARSE_ERROR,
)
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from sqlalchemy import create_engine

# ── Colour helpers ─────────────────────────────────────────────────────────────
GRN  = "\033[92m"
RED  = "\033[91m"
YLW  = "\033[93m"
BLU  = "\033[94m"
BOLD = "\033[1m"
RST  = "\033[0m"

def _ok(msg):  return f"{GRN}✅ {msg}{RST}"
def _fail(msg): return f"{RED}❌ {msg}{RST}"
def _warn(msg): return f"{YLW}⚠️  {msg}{RST}"


# ── App configs ────────────────────────────────────────────────────────────────

FOOTBALL_DB = os.path.join(BASE, "Football Agent", "database.sqlite")
NBA_DB      = os.path.join(BASE, "Nba Agent 2", "NBA Data", "nba.sqlite")

FOOTBALL_CONTEXT = (
    "European football statistics — 11 leagues (Premier League, La Liga, Bundesliga, "
    "Serie A, etc.), 25,979 matches, 11,060 players, seasons 2008 to 2016. "
    "Tables: Match, Team, Player, Player_Attributes, League. "
    "Enrichment (Transfermarkt, 2008-2016): player_valuations and player_appearances."
)

NBA_CONTEXT = (
    "NBA basketball statistics — 60,192 regular season games, 4,815 players, "
    "30 teams, seasons 1946 to 2023. "
    "Tables: game, player, team, common_player_info, line_score. "
    "Enrichment: player_salary (annual salaries 2000-2025) joined via player_name_lookup."
)

FOOTBALL_SCHEMA_HINTS = """
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

NBA_SCHEMA_HINTS = """
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

Draft and college info (in common_player_info):
- school: the college/university the player attended (e.g. 'Duke', 'Kentucky', 'None' if pro/international).
- country: the player's nationality (e.g. 'USA', 'France'). NOT the school's country.
- To count players who went to school outside the US, filter school IS NOT NULL AND school != '' AND school != 'None'
  AND country != 'USA'. Do NOT use country != 'USA' alone — that filters by nationality, not school location.

IMPORTANT — to get total points scored by a team across games, you MUST use UNION ALL to combine
home and away contributions. NEVER sum only pts_home or only pts_away — that undercounts by ~50%.
Correct pattern:
  SELECT team_name, SUM(pts) AS total_points FROM (
      SELECT team_name_home AS team_name, pts_home AS pts FROM game
      WHERE season_id LIKE '2%' AND pts_home IS NOT NULL
      UNION ALL
      SELECT team_name_away AS team_name, pts_away AS pts FROM game
      WHERE season_id LIKE '2%' AND pts_away IS NOT NULL
  ) GROUP BY team_name ORDER BY total_points DESC;
This applies to any "total points by team" question, with or without a season filter.
To filter by season year: season_id LIKE '2YYYY%' (e.g. '22021%' for 2021-22).

IMPORTANT — there is NO per-player scoring column in this database. The game table only has
team-level totals (pts_home, pts_away). There is NO box score table with individual player points.
If asked for top scorers by total points, return:
  SELECT 'Individual player scoring totals are not available in this dataset — only team-level points exist.' AS answer;

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

FOOTBALL_ENTITY_TABLES = [
    ("Team",   "team_long_name"),
    ("Player", "player_name"),
    ("League", "name"),
]

NBA_ENTITY_TABLES = [
    ("team",               "full_name"),
    ("player",             "full_name"),
    ("common_player_info", "display_first_last"),
]

FOOTBALL_QUESTIONS = [
    "Which team scored the most goals across all seasons?",
    "Who are the top 5 players by overall FIFA rating?",
    "Which league had the highest average goals per match?",
    "How many matches ended in a draw in the 2015/2016 Premier League?",
    "What was FC Barcelona's win rate at home vs away?",
    "Which players improved their rating the most between 2010 and 2015?",
    "Do taller players tend to have better overall ratings?",
    "Which season had the most total goals scored?",
    "Who were the 10 most valuable players in 2013?",
    "Which players scored the most real goals between 2010 and 2014?",
    "How often did the Bet365 favorite actually win the match?",
    "Which team won the most matches as the underdog?",
]

NBA_QUESTIONS = [
    "Which team scored the most total points in NBA history?",
    "Which team has the most wins in NBA history?",
    "Which team scored the most points in the 2021-22 season?",
    "Which team had the best win rate in the 2020-21 regular season?",
    "What is the average height of NBA players by position?",
    "How many players in the dataset attended a college outside the US?",
    "What is the average points per game by season?",
    "How many games went to overtime?",
    "Who were the 10 highest-paid players in the 2021 season?",
    "How has the average player salary changed over the years?",
]

# ── Setup resources ────────────────────────────────────────────────────────────

def build_resources(db_path: str, include_tables: list):
    ro_uri = f"sqlite:///file:{db_path}?mode=ro&uri=true"
    engine = create_engine(ro_uri)
    db = SQLDatabase.from_uri(ro_uri, include_tables=include_tables)
    llm = ChatOpenAI(
        base_url="https://api.deepseek.com", model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"], temperature=0,
    )
    agent = create_sql_agent(
        llm=llm, db=db, handle_parsing_errors=True,
        max_iterations=12, verbose=False, return_intermediate_steps=True,
    )
    schema = db.get_table_info()
    return engine, llm, agent, schema


# ── Test runner ────────────────────────────────────────────────────────────────

STATE_LABEL = {
    SUCCESS:      "SUCCESS",
    SQL_ERROR:    "SQL_ERROR",
    EMPTY_RESULT: "EMPTY_RESULT",
    TOO_COMPLEX:  "TOO_COMPLEX",
    PARSE_ERROR:  "PARSE_ERROR",
}

def run_suite(app_name: str, questions: list, db_path: str, include_tables: list,
              dataset_context: str, schema_hints: str, entity_tables: list):
    print(f"\n{BOLD}{BLU}{'='*64}{RST}")
    print(f"{BOLD}{BLU}  {app_name}{RST}")
    print(f"{BOLD}{BLU}{'='*64}{RST}\n")

    print("  Building resources (engine, LLM, agent)...", end=" ", flush=True)
    engine, llm, agent, schema = build_resources(db_path, include_tables)
    print("done\n")

    results = []
    for i, question in enumerate(questions, 1):
        print(f"  [{i:02d}/{len(questions)}] {question}")
        tokens = fresh_tokens()
        t0 = time.time()

        route = route_question(question, llm, tokens, dataset_context)
        route_label = {"SIMPLE": "⚡SIMPLE", "ENTITY": "🔍ENTITY",
                       "COMPLEX": "🔄COMPLEX", "OUT_OF_SCOPE": "🚫OOS"}.get(route, route)

        if route == "OUT_OF_SCOPE":
            state, sql, df, answer = "out_of_scope", None, None, None
        elif route == "SIMPLE":
            state, sql, df, answer = single_shot(
                question, engine, llm, schema, tokens, schema_hints)
        elif route == "ENTITY":
            hints = resolve_entities(question, engine, entity_tables)
            state, sql, df, answer = single_shot(
                question, engine, llm, schema, tokens, schema_hints, extra_hints=hints)
        else:  # COMPLEX
            state, sql, df, answer = react(question, agent, engine, tokens)

        elapsed = time.time() - t0
        passed  = state == SUCCESS

        status_str = _ok(f"{STATE_LABEL.get(state, state)}") if passed else _fail(f"{STATE_LABEL.get(state, state)}")
        print(f"         route={route_label}  state={status_str}  {elapsed:.1f}s")

        if passed and df is not None and not df.empty:
            preview = df.head(3).to_string(index=False, max_colwidth=40)
            for line in preview.splitlines():
                print(f"         {line}")
        elif not passed:
            detail = (answer or "")[:160]
            print(f"         {_warn(detail)}")
        print()

        results.append({
            "question": question,
            "route": route,
            "state": state,
            "elapsed": elapsed,
            "passed": passed,
        })

    passed_n = sum(r["passed"] for r in results)
    total_n  = len(results)
    print(f"  {BOLD}Result: {passed_n}/{total_n} passed{RST}")
    failed = [r for r in results if not r["passed"]]
    if failed:
        print(f"  {RED}Failed questions:{RST}")
        for r in failed:
            print(f"    • {r['question']}  [{STATE_LABEL.get(r['state'], r['state'])}]")
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results = []

    football_results = run_suite(
        app_name      = "Statline · Football",
        questions     = FOOTBALL_QUESTIONS,
        db_path       = FOOTBALL_DB,
        include_tables= ["Match", "Player", "Team", "League", "Player_Attributes",
                         "player_valuations", "player_appearances"],
        dataset_context = FOOTBALL_CONTEXT,
        schema_hints  = FOOTBALL_SCHEMA_HINTS,
        entity_tables = FOOTBALL_ENTITY_TABLES,
    )

    nba_results = run_suite(
        app_name      = "Statline · Hoops",
        questions     = NBA_QUESTIONS,
        db_path       = NBA_DB,
        include_tables= ["game", "player", "team", "common_player_info", "line_score",
                         "player_salary", "player_name_lookup"],
        dataset_context = NBA_CONTEXT,
        schema_hints  = NBA_SCHEMA_HINTS,
        entity_tables = NBA_ENTITY_TABLES,
    )

    all_results = football_results + nba_results
    total   = len(all_results)
    passed  = sum(r["passed"] for r in all_results)
    failed  = total - passed

    print(f"\n{BOLD}{'='*64}{RST}")
    print(f"{BOLD}  OVERALL: {passed}/{total} passed", end="")
    if failed:
        print(f"  {RED}({failed} FAILED){RST}")
    else:
        print(f"  {GRN}— all clear{RST}")
    print(f"{BOLD}{'='*64}{RST}\n")
