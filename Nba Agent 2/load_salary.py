"""
load_salary.py
Loads NBA Player Salaries_2000-2025.csv into nba.sqlite as player_salary.
Also creates player_name_lookup (normalised names for both sides of the join).
Run once: python3 load_salary.py
"""
import sqlite3, re, unicodedata
import pandas as pd

DB_PATH  = "NBA Data/nba.sqlite"
CSV_PATH = "NBA Data/csv/NBA Player Salaries_2000-2025.csv"


# Known spelling variants where two legitimate romanisations exist and can't
# be resolved by character normalisation alone. Key = salary-side form after
# normalisation, value = DB-side form after normalisation.
ALIASES = {
    "dennis schroeder": "dennis schroder",
}


def normalise(name: str) -> str:
    """Canonical form used for joining — applied to both salary and DB player names."""
    # 1. Strip accents: Dončić → Doncic, Jokić → Jokic
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    # 2. Remove Jr./Sr./II/III/IV suffixes (with or without trailing period)
    name = re.sub(r"\s+(Jr\.?|Sr\.?|II|III|IV|V)\s*$", "", name, flags=re.I)
    # 3. Remove dots after single letters: P.J. → PJ, C.J. → CJ, A.J. → AJ
    name = re.sub(r"\b([A-Za-z])\.", r"\1", name)
    # 4. Lowercase and collapse whitespace
    result = re.sub(r"\s+", " ", name).strip().lower()
    # 5. Apply manual aliases for irreconcilable spelling variants
    return ALIASES.get(result, result)


# ── Load and normalise salary CSV ─────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)
df = df.rename(columns={"Player": "player_name", "Salary": "salary", "Season": "season_start_year"})
df["player_name_normalised"] = df["player_name"].apply(normalise)
df = df[["player_name", "player_name_normalised", "salary", "season_start_year"]]

# ── Load and normalise player names from DB ───────────────────────────────────

conn = sqlite3.connect(DB_PATH)
players = pd.read_sql("SELECT id AS player_id, full_name FROM player", conn)
players["full_name_normalised"] = players["full_name"].apply(normalise)

# ── Write both tables ─────────────────────────────────────────────────────────

df.to_sql("player_salary", conn, if_exists="replace", index=False)
players[["player_id", "full_name", "full_name_normalised"]].to_sql(
    "player_name_lookup", conn, if_exists="replace", index=False
)
print(f"player_salary:      {len(df):,} rows written")
print(f"player_name_lookup: {len(players):,} rows written")
print()

# ── Join coverage ─────────────────────────────────────────────────────────────

exact = pd.read_sql("""
    SELECT COUNT(*) AS n FROM player_salary ps
    JOIN player p ON p.full_name = ps.player_name
""", conn).iloc[0, 0]

normalised = pd.read_sql("""
    SELECT COUNT(*) AS n FROM player_salary ps
    JOIN player_name_lookup pnl ON pnl.full_name_normalised = ps.player_name_normalised
""", conn).iloc[0, 0]

total = len(df)
print("── Join coverage ─────────────────────────────────────────────────────")
print(f"  Total salary rows:             {total:>6,}")
print(f"  Exact name join:               {exact:>6,}  ({exact/total*100:.1f}%)")
print(f"  Normalised name join:          {normalised:>6,}  ({normalised/total*100:.1f}%)")
print(f"  Still unmatched:               {total-normalised:>6,}  ({(total-normalised)/total*100:.1f}%)")
print()

# ── Previously failing names — did they get fixed? ────────────────────────────

checks = ["PJ Tucker", "Tim Hardaway Jr", "Kelly Oubre", "Otto Porter",
          "Dennis Schroeder", "Luka Doncic", "Nikola Jokic"]

print("── Spot-check previously failing names ──────────────────────────────")
for name in checks:
    norm = normalise(name)
    hit = pd.read_sql(
        "SELECT full_name FROM player_name_lookup WHERE full_name_normalised = ?",
        conn, params=(norm,)
    )
    status = f"✅  → {hit['full_name'].iloc[0]}" if not hit.empty else "❌  no match"
    print(f"  {name:<28} {status}")
print()

# ── Remaining unmatched — sample ──────────────────────────────────────────────

unmatched = pd.read_sql("""
    SELECT DISTINCT ps.player_name, ps.player_name_normalised
    FROM player_salary ps
    LEFT JOIN player_name_lookup pnl ON pnl.full_name_normalised = ps.player_name_normalised
    WHERE pnl.player_id IS NULL
    ORDER BY ps.player_name
    LIMIT 20
""", conn)
print("── Remaining unmatched sample ────────────────────────────────────────")
print(unmatched.to_string(index=False))
print()

# ── Smoke test — top earners 2021 via normalised join ─────────────────────────

top = pd.read_sql("""
    SELECT p.full_name, ps.salary, ps.season_start_year
    FROM player_salary ps
    JOIN player_name_lookup pnl ON pnl.full_name_normalised = ps.player_name_normalised
    JOIN player p ON p.id = pnl.player_id
    WHERE ps.season_start_year = 2021
    ORDER BY ps.salary DESC
    LIMIT 10
""", conn)
print("── Top 10 salaries 2021 (normalised join) ────────────────────────────")
print(top.to_string(index=False))

conn.close()
print("\nDone.")
