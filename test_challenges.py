import os, re
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from sqlalchemy import text

FOOTBALL_DB = "Football Agent/database.sqlite"
NBA_DB      = "Nba Agent 2/NBA Data/nba.sqlite"

FOOTBALL_HINTS = """
Key join: Player.player_api_id = Player_Attributes.player_api_id
Key join: Match.home_team_api_id = Team.team_api_id (and away_team_api_id)
Key join: Match.league_id = League.id
Goals scored by a team = home_team_goal (when home) + away_team_goal (when away) — use UNION ALL, not OR join.
Player ratings are in Player_Attributes (overall_rating, potential, finishing, etc).
"""

NBA_HINTS = """
Key tables and what they contain:
- game: one row per game, has both home and away stats in the same row.
  Points: pts_home, pts_away. Win/loss: wl_home and wl_away ('W' or 'L').
  Team names are already in the table: team_name_home, team_name_away — no join needed for basic game queries.
  season_id format: '22021' = 2021-22 regular season (first digit is season type, 2 = regular season).
- player: basic info only (id, full_name, is_active). For height, weight, position, draft info use common_player_info.
- common_player_info: detailed player profiles. Join: player.id = common_player_info.person_id.
- team: full team details. Join: game.team_id_home = team.id or game.team_id_away = team.id.
- line_score: quarter-by-quarter scores per game. Join: line_score.game_id = game.game_id.
To get total points scored by a team across games, SUM both pts_home (when they were home) and pts_away (when they were away) using UNION ALL.
To filter by season year, use: season_id LIKE '2YYYY%' where YYYY is the season start year (e.g. '22021%' for 2021-22).
"""

FOOTBALL_QUESTIONS = [
    ("Multi-table join",       "Which league has the most players with an overall rating above 85?"),
    ("Self-join / YoY",        "Which players improved their overall rating the most between 2012 and 2015?"),
    ("Aggregate then filter",  "Which teams scored more than 100 home goals in a single season?"),
    ("Schema trap",            "What is the head-to-head record between Real Madrid and Barcelona?"),
]

NBA_QUESTIONS = [
    ("Multi-table join",       "Which drafted players from outside the USA had the highest win rates?"),
    ("YoY comparison",         "Which teams improved their win rate the most from the 2018-19 to 2019-20 season?"),
    ("Aggregate then filter",  "Which teams averaged more than 115 points per game across an entire season?"),
    ("Schema trap",            "How many games went to overtime?"),
]

def make_db_and_llm(db_path, tables):
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}", include_tables=tables)
    llm = ChatOpenAI(
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0
    )
    return db, llm

def ask(question, db, llm, hints):
    schema = db.get_table_info()
    prompt = f"""You are a SQL expert. Given the schema and hints below, write a SQLite query to answer the question.

Schema:
{schema}

Hints:
{hints}

Question: {question}

Reply in this exact format:
SQL:
```sql
<your query here>
```
Answer: <one sentence explaining what the result shows>"""

    response = llm.invoke(prompt).content
    match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL)
    if not match:
        return None, None, "Could not parse SQL"

    sql = match.group(1).strip()
    answer_line = re.search(r"Answer:\s*(.+)", response)
    explanation = answer_line.group(1).strip() if answer_line else ""

    try:
        with db._engine.connect() as conn:
            df = pd.read_sql_query(text(sql), conn)
        return sql, df, explanation
    except Exception as e:
        return sql, None, f"SQL ERROR: {e}"

def run_tests(label, db_path, tables, hints, questions):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    db, llm = make_db_and_llm(db_path, tables)

    for category, question in questions:
        print(f"\n[{category}]")
        print(f"Q: {question}")
        sql, df, answer = ask(question, db, llm, hints)

        if sql:
            print(f"SQL:\n{sql}")
        if df is not None:
            print(f"Result ({len(df)} rows):\n{df.to_string(index=False)}")
        else:
            print(f"Result: None")
        print(f"Answer: {answer}")
        print(f"{'-'*70}")

run_tests(
    "FOOTBALL AGENT",
    FOOTBALL_DB,
    ["Match", "Player", "Team", "League", "Player_Attributes"],
    FOOTBALL_HINTS,
    FOOTBALL_QUESTIONS
)

run_tests(
    "NBA AGENT",
    NBA_DB,
    ["game", "player", "team", "common_player_info", "line_score"],
    NBA_HINTS,
    NBA_QUESTIONS
)
