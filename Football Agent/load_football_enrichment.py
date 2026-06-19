"""
load_football_enrichment.py
Loads davidcariboo player valuations and match appearances into database.sqlite.
Creates three tables:
  - enrichment_player_bridge   : maps player_api_id (DB) <-> cariboo_player_id
  - player_valuations          : historical market values 2008-2016
  - player_appearances         : per-match goals/assists/cards 2008-2016
Run once: python3 load_football_enrichment.py
"""
import sqlite3, re, unicodedata
import pandas as pd

DB_PATH    = "database.sqlite"
ARCHIVE    = "/Users/macbookair/Downloads/archive-4"

# ── Spelling variants where normalisation alone can't close the gap ───────────
# Key = football DB name after normalise(), value = cariboo name after normalise()
ALIASES = {
    "eric maxim choupo-moting": "eric-maxim choupo-moting",  # hyphen position
    "ilkay guendogan":          "ilkay gundogan",             # ue vs u
    "henrik mkhitaryan":        "henrikh mkhitaryan",         # Henrik vs Henrikh
    "mehdi benatia":            "medhi benatia",               # letter swap
    "phillippe mexes":          "philippe mexes",              # double-l
    "lasse schoene":            "lasse schone",                # oe vs o
    "sulley ali muntari":       "sulley muntari",              # middle name
    "christian guenter":        "christian gunter",            # ue vs u
}


def normalise(name: str) -> str:
    """Canonical form applied to both sides of the join."""
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"\s+(Jr\.?|Sr\.?|II|III|IV|V)\s*$", "", name, flags=re.I)
    name = re.sub(r"\b([A-Za-z])\.", r"\1", name)
    return re.sub(r"\s+", " ", name).strip().lower()


def _german_fix(norm: str) -> str:
    """Convert French-style umlaut romanization (oe/ue/ae) to plain ASCII."""
    n = re.sub(r"(?<=[a-z])oe(?=[a-z])", "o", norm)
    n = re.sub(r"(?<=[a-z])ue(?=[a-z])", "u", n)
    n = re.sub(r"(?<=[a-z])ae(?=[a-z])", "a", n)
    return n


def _first_last(norm: str) -> str:
    """Strip middle names — keep first and last token only."""
    parts = norm.split()
    return (parts[0] + " " + parts[-1]) if len(parts) > 2 else norm


def _dehyphen(norm: str) -> str:
    return norm.replace("-", " ")


def find_match(db_norm: str, cariboo_set: set) -> str | None:
    """Return the matching cariboo norm via a priority chain of fallbacks."""
    # 1. Exact
    if db_norm in cariboo_set:
        return db_norm
    # 2. Manual alias
    alias = ALIASES.get(db_norm)
    if alias and alias in cariboo_set:
        return alias
    # 3. German romanization fix
    g = _german_fix(db_norm)
    if g != db_norm and g in cariboo_set:
        return g
    # 4. First + last name only
    fl = _first_last(db_norm)
    if fl != db_norm and fl in cariboo_set:
        return fl
    # 5. Hyphen -> space
    dh = _dehyphen(db_norm)
    if dh != db_norm and dh in cariboo_set:
        return dh
    # 6. German fix + first+last combined
    gfl = _first_last(g)
    if gfl != db_norm and gfl in cariboo_set:
        return gfl
    return None


# ── Load sources ──────────────────────────────────────────────────────────────

print("Loading sources…")
conn = sqlite3.connect(DB_PATH)
db_players = pd.read_sql("SELECT player_api_id, player_name FROM Player", conn)
db_players["db_norm"] = db_players["player_name"].apply(normalise)

cariboo_players = pd.read_csv(f"{ARCHIVE}/players.csv")
cariboo_players["norm"] = cariboo_players["name"].apply(normalise)
cariboo_set    = set(cariboo_players["norm"])
cariboo_id_map = dict(zip(cariboo_players["norm"], cariboo_players["player_id"]))

# ── Build bridge ──────────────────────────────────────────────────────────────

print("Building player bridge…")
rows = []
for _, r in db_players.iterrows():
    matched_norm = find_match(r["db_norm"], cariboo_set)
    if matched_norm:
        method = (
            "exact"       if matched_norm == r["db_norm"]                     else
            "alias"       if r["db_norm"] in ALIASES                           else
            "german_fix"  if _german_fix(r["db_norm"]) == matched_norm        else
            "first_last"  if _first_last(r["db_norm"]) == matched_norm        else
            "dehyphen"    if _dehyphen(r["db_norm"]) == matched_norm           else
            "german+fl"
        )
        rows.append({
            "player_api_id":    r["player_api_id"],
            "db_player_name":   r["player_name"],
            "cariboo_player_id": cariboo_id_map[matched_norm],
            "match_method":     method,
        })

bridge = pd.DataFrame(rows)
total_db = len(db_players)
matched  = len(bridge)

print(f"\n── Bridge coverage ───────────────────────────────────────")
print(f"  Total football DB players:    {total_db:>6,}")
print(f"  Matched to cariboo:           {matched:>6,}  ({matched/total_db*100:.1f}%)")
print(f"  Unmatched:                    {total_db-matched:>6,}  ({(total_db-matched)/total_db*100:.1f}%)")
method_counts = bridge["match_method"].value_counts()
for method, count in method_counts.items():
    print(f"    {method:<20} {count:>5,}")

# ── Filter and load player_valuations ────────────────────────────────────────

print("\nLoading player_valuations (2008-2016)…")
pv = pd.read_csv(f"{ARCHIVE}/player_valuations.csv")
pv["year"] = pv["date"].str[:4].astype(int)
pv_era = pv[pv["year"].between(2008, 2016)].copy()
pv_era = pv_era.rename(columns={"player_id": "cariboo_player_id"})
pv_era = pv_era.merge(bridge[["cariboo_player_id","player_api_id"]], on="cariboo_player_id", how="inner")
pv_era = pv_era[["player_api_id","cariboo_player_id","date","market_value_in_eur","current_club_name"]]
pv_era = pv_era.sort_values(["player_api_id","date"])

# ── Filter and load player_appearances ───────────────────────────────────────

print("Loading player_appearances (2008-2016)…")
ap = pd.read_csv(f"{ARCHIVE}/appearances.csv")
ap["year"] = ap["date"].str[:4].astype(int)
ap_era = ap[ap["year"].between(2008, 2016)].copy()
ap_era = ap_era.rename(columns={"player_id": "cariboo_player_id"})
ap_era = ap_era.merge(bridge[["cariboo_player_id","player_api_id"]], on="cariboo_player_id", how="inner")
ap_era = ap_era[[
    "player_api_id","cariboo_player_id","game_id","date","competition_id",
    "yellow_cards","red_cards","goals","assists","minutes_played"
]]
ap_era = ap_era.sort_values(["player_api_id","date"])

# ── Write tables ──────────────────────────────────────────────────────────────

print("\nWriting tables to database.sqlite…")
bridge.to_sql("enrichment_player_bridge", conn, if_exists="replace", index=False)
pv_era.to_sql("player_valuations", conn, if_exists="replace", index=False)
ap_era.to_sql("player_appearances", conn, if_exists="replace", index=False)
conn.commit()

print(f"  enrichment_player_bridge: {len(bridge):,} rows")
print(f"  player_valuations:        {len(pv_era):,} rows  ({pv_era['player_api_id'].nunique():,} players)")
print(f"  player_appearances:       {len(ap_era):,} rows  ({ap_era['player_api_id'].nunique():,} players)")

# ── Spot checks ───────────────────────────────────────────────────────────────

print("\n── Spot checks ───────────────────────────────────────────")
checks = [
    "Lionel Messi", "Cristiano Ronaldo", "Wayne Rooney",
    "Zlatan Ibrahimovic", "Andres Iniesta", "Eden Hazard",
    "Gareth Bale", "Neymar", "Luis Suarez",
    "Ilkay Guendogan", "Henrik Mkhitaryan", "Mehdi Benatia",
    "Eric Maxim Choupo-Moting", "Sulley Ali Muntari", "Phillippe Mexes",
]
for name in checks:
    n = normalise(name)
    row = bridge[bridge["db_player_name"] == name]
    if row.empty:
        print(f"  {name:<32} ❌  not in football DB")
        continue
    api_id = row.iloc[0]["player_api_id"]
    method = row.iloc[0]["match_method"]
    val = pv_era[pv_era["player_api_id"] == api_id]
    if val.empty:
        print(f"  {name:<32} ✅  matched ({method}), no valuation data")
    else:
        best = val.nlargest(1, "market_value_in_eur").iloc[0]
        print(f"  {name:<32} ✅  peak €{best['market_value_in_eur']/1e6:.0f}M in {best['date'][:4]}  [{method}]")

# ── Smoke test query ──────────────────────────────────────────────────────────

print("\n── Top 10 highest-valued players (2013, all time peak) ───")
top = pd.read_sql("""
    SELECT p.player_name,
           MAX(pv.market_value_in_eur) AS peak_value,
           pv.current_club_name
    FROM player_valuations pv
    JOIN Player p ON p.player_api_id = pv.player_api_id
    WHERE pv.date LIKE '2013%'
    GROUP BY p.player_api_id
    ORDER BY peak_value DESC
    LIMIT 10
""", conn)
for _, r in top.iterrows():
    print(f"  {r['player_name']:<28} €{r['peak_value']/1e6:.0f}M  ({r['current_club_name']})")

print("\n── Most goals in appearances (2010-2014) ─────────────────")
goals = pd.read_sql("""
    SELECT p.player_name,
           SUM(pa.goals)   AS total_goals,
           SUM(pa.assists) AS total_assists,
           COUNT(*)        AS matches
    FROM player_appearances pa
    JOIN Player p ON p.player_api_id = pa.player_api_id
    WHERE pa.date BETWEEN '2010-01-01' AND '2014-12-31'
    GROUP BY pa.player_api_id
    ORDER BY total_goals DESC
    LIMIT 10
""", conn)
print(goals.to_string(index=False))

conn.close()
print("\nDone.")
