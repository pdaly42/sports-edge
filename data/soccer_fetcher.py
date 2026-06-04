"""
International football data pipeline for World Cup model.

Data sources (all free, no API key):
  - ESPN API: World Cup (2002–2022), UEFA Euro, AFCON, UEFA Nations League
  - StatsBomb open data: WC 2018 & 2022 (supplemental)

Pipeline:
  1. Fetch results from ESPN across all tournaments
  2. Compute rolling ELO ratings chronologically
  3. Build per-game matchup features
  4. Return matchup DataFrame for model training
"""

import sys, os, warnings, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import numpy as np
from pathlib import Path

from config.settings import RAW_DIR

SOCCER_RAW = RAW_DIR / "soccer"
SOCCER_RAW.mkdir(parents=True, exist_ok=True)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"

# All tournaments to collect, with (league_slug, date_range, tournament_label, importance_weight)
# importance_weight drives ELO K-factor: higher = bigger rating changes
TOURNAMENTS = [
    # World Cups (most important)
    ("fifa.world", "20020531", "20020630", "World Cup", 60),
    ("fifa.world", "20060609", "20060709", "World Cup", 60),
    ("fifa.world", "20100611", "20100711", "World Cup", 60),
    ("fifa.world", "20140612", "20140713", "World Cup", 60),
    ("fifa.world", "20180614", "20180715", "World Cup", 60),
    ("fifa.world", "20221101", "20221218", "World Cup", 60),
    # UEFA Euros
    ("uefa.euro", "20120608", "20120701", "UEFA Euro",  50),
    ("uefa.euro", "20160610", "20160710", "UEFA Euro",  50),
    ("uefa.euro", "20210611", "20210711", "UEFA Euro",  50),
    ("uefa.euro", "20240614", "20240714", "UEFA Euro",  50),
    # AFCON
    ("caf.nations", "20190621", "20190719", "AFCON",    40),
    ("caf.nations", "20220109", "20220206", "AFCON",    40),
    ("caf.nations", "20240113", "20240211", "AFCON",    40),
    # UEFA Nations League (recent form signal)
    ("uefa.nations", "20221009", "20221016", "Nations League", 30),
    ("uefa.nations", "20230609", "20230620", "Nations League", 30),
    ("uefa.nations", "20241010", "20241015", "Nations League", 30),
]

# Odds API team name → ESPN team name (for live predictions)
ODDS_TO_ESPN = {
    "United States": "USA",
    "South Korea": "Korea Republic",
    "Czech Republic": "Czechia",
    "Ivory Coast": "Côte d'Ivoire",
    "Iran": "IR Iran",
    "North Korea": "Korea DPR",
    "Cape Verde": "Cabo Verde",
    "Congo DR": "DR Congo",
    "Trinidad & Tobago": "Trinidad and Tobago",
    "Bosnia & Herzegovina": "Bosnia-Herzegovina",
}

# ELO starting values for seeding (better than flat 1500)
ELO_SEEDS = {
    "Brazil": 2000, "France": 1990, "England": 1970, "Germany": 1970,
    "Spain": 1960, "Argentina": 1980, "Portugal": 1940, "Netherlands": 1940,
    "Italy": 1930, "Belgium": 1920, "Croatia": 1890, "Uruguay": 1880,
    "Mexico": 1850, "Colombia": 1850, "Denmark": 1860, "Switzerland": 1850,
    "USA": 1820, "Senegal": 1820, "Morocco": 1830, "Japan": 1820,
    "South Korea": 1810, "Australia": 1800, "Serbia": 1830, "Poland": 1820,
    "Sweden": 1850, "Norway": 1820, "Austria": 1810, "Czech Republic": 1820,
    "Czechia": 1820, "Hungary": 1800, "Romania": 1790, "Slovakia": 1790,
    "Ghana": 1800, "Nigeria": 1810, "Cameroon": 1800, "Egypt": 1800,
    "Algeria": 1800, "Tunisia": 1790, "Ecuador": 1800, "Chile": 1840,
    "Peru": 1810, "Paraguay": 1790, "Bolivia": 1750, "Venezuela": 1760,
    "Costa Rica": 1800, "Honduras": 1770, "Panama": 1770, "Jamaica": 1760,
    "Canada": 1820, "New Zealand": 1750, "Saudi Arabia": 1780, "Qatar": 1760,
    "Iran": 1790, "Korea Republic": 1810, "IR Iran": 1790,
}

DEFAULT_ELO = 1750


# ─────────────────────────────────────────────────────────────
# ESPN Data Fetcher
# ─────────────────────────────────────────────────────────────

def fetch_espn_events(league: str, date_from: str, date_to: str) -> list:
    """Fetch all completed match events from ESPN for a date range."""
    url = f"{ESPN_BASE}/{league}/scoreboard"
    params = {"dates": f"{date_from}-{date_to}"}
    try:
        r = requests.get(url, params=params, timeout=12)
        if r.status_code != 200:
            return []
        return r.json().get("events", [])
    except Exception as e:
        print(f"    ESPN error ({league} {date_from}): {e}")
        return []


def parse_event(event: dict, tournament: str):
    """Extract key fields from an ESPN event object."""
    try:
        comp = event["competitions"][0]
        # Skip if not completed
        if not comp.get("status", {}).get("type", {}).get("completed", False):
            return None

        teams = comp["competitors"]
        # ESPN: homeAway field tells us which is home
        home = next((t for t in teams if t.get("homeAway") == "home"), teams[0])
        away = next((t for t in teams if t.get("homeAway") == "away"), teams[1])

        home_score = int(home.get("score", 0))
        away_score = int(away.get("score", 0))
        home_name  = home["team"].get("displayName") or home["team"].get("name")
        away_name  = away["team"].get("displayName") or away["team"].get("name")

        date = event.get("date", "")[:10]
        if not date or not home_name or not away_name:
            return None

        # 1 = home win, 0 = draw, -1 = away win
        if home_score > away_score:
            result = 1
        elif home_score == away_score:
            result = 0
        else:
            result = -1

        # All major international tournaments are effectively neutral venues
        neutral = True

        return {
            "date":       date,
            "tournament": tournament,
            "home_team":  home_name,
            "away_team":  away_name,
            "home_score": home_score,
            "away_score": away_score,
            "result":     result,  # 1=home, 0=draw, -1=away
            "neutral":    neutral,
            "goal_diff":  abs(home_score - away_score),
        }
    except Exception:
        return None


def fetch_all_matches(force_refresh: bool = False) -> pd.DataFrame:
    """Fetch and cache all international matches across configured tournaments."""
    cache = SOCCER_RAW / "all_matches.csv"
    if cache.exists() and not force_refresh:
        print("  Loading cached international match data")
        return pd.read_csv(cache, parse_dates=["date"])

    print("  Fetching international match data from ESPN...")
    all_rows = []

    for league, date_from, date_to, tournament, _ in TOURNAMENTS:
        print(f"    {tournament} {date_from[:4]}...", end=" ")
        events = fetch_espn_events(league, date_from, date_to)
        rows = [r for e in events if (r := parse_event(e, tournament)) is not None]
        all_rows.extend(rows)
        print(f"{len(rows)} matches")
        time.sleep(0.3)  # gentle throttle

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(
        subset=["date", "home_team", "away_team"]
    ).reset_index(drop=True)

    df.to_csv(cache, index=False)
    print(f"  Saved {len(df)} total matches")
    return df


# ─────────────────────────────────────────────────────────────
# ELO Engine
# ─────────────────────────────────────────────────────────────

def expected_score(elo_a: float, elo_b: float) -> float:
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def goal_diff_multiplier(gd: int, won: bool) -> float:
    """Standard World Football ELO goal-difference multiplier."""
    if not won:
        return 1.0
    if gd == 1:
        return 1.0
    elif gd == 2:
        return 1.5
    elif gd == 3:
        return 1.75
    else:
        return 1.75 + (gd - 3) / 8


def compute_elo(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Add pre-match ELO ratings to each game row and return updated ratings dict.
    Modifies nothing in-place; returns enriched DataFrame + final ELO snapshot.
    """
    elo = {}  # team → current elo

    # K-factor by tournament
    k_map = {"World Cup": 60, "UEFA Euro": 50, "AFCON": 40, "Nations League": 30}

    rows = []
    for _, m in matches.sort_values("date").iterrows():
        h, a = m["home_team"], m["away_team"]
        elo_h = elo.get(h, ELO_SEEDS.get(h, DEFAULT_ELO))
        elo_a = elo.get(a, ELO_SEEDS.get(a, DEFAULT_ELO))

        exp_h = expected_score(elo_h, elo_a)
        exp_a = 1 - exp_h

        result = m["result"]  # 1=home, 0=draw, -1=away
        actual_h = 1.0 if result == 1 else (0.5 if result == 0 else 0.0)
        actual_a = 1 - actual_h

        K  = k_map.get(m["tournament"], 30)
        gd = int(m["goal_diff"])
        gd_mult_h = goal_diff_multiplier(gd, result == 1)
        gd_mult_a = goal_diff_multiplier(gd, result == -1)

        new_elo_h = elo_h + K * gd_mult_h * (actual_h - exp_h)
        new_elo_a = elo_a + K * gd_mult_a * (actual_a - exp_a)

        row = m.to_dict()
        row["home_elo_pre"]  = round(elo_h, 1)
        row["away_elo_pre"]  = round(elo_a, 1)
        row["elo_diff"]      = round(elo_h - elo_a, 1)

        elo[h] = new_elo_h
        elo[a] = new_elo_a
        rows.append(row)

    return pd.DataFrame(rows), elo


# ─────────────────────────────────────────────────────────────
# Rolling form features
# ─────────────────────────────────────────────────────────────

def add_rolling_form(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Add rolling pre-match form features (win%, goals for/against avg) per team.
    Works from both the home and away team's perspective.
    """
    # Build a flat per-team game log first
    home_log = df[["date","home_team","away_team","home_score","away_score","result"]].copy()
    home_log.columns = ["date","team","opponent","gf","ga","result_raw"]
    home_log["win"]  = (home_log["result_raw"] == 1).astype(int)
    home_log["draw"] = (home_log["result_raw"] == 0).astype(int)
    home_log["gd"]   = home_log["gf"] - home_log["ga"]

    away_log = df[["date","away_team","home_team","away_score","home_score","result"]].copy()
    away_log.columns = ["date","team","opponent","gf","ga","result_raw"]
    away_log["win"]  = (away_log["result_raw"] == -1).astype(int)
    away_log["draw"] = (away_log["result_raw"] == 0).astype(int)
    away_log["gd"]   = away_log["gf"] - away_log["ga"]

    log = pd.concat([home_log, away_log]).sort_values(["team","date"]).reset_index(drop=True)

    def rolling_shifted(s, w):
        return s.shift(1).rolling(w, min_periods=3).mean()

    log[f"win_pct_{window}g"]  = log.groupby("team")["win"].transform(lambda x: rolling_shifted(x, window))
    log[f"gf_avg_{window}g"]   = log.groupby("team")["gf"].transform(lambda x:  rolling_shifted(x, window))
    log[f"ga_avg_{window}g"]   = log.groupby("team")["ga"].transform(lambda x:  rolling_shifted(x, window))
    log[f"gd_avg_{window}g"]   = log.groupby("team")["gd"].transform(lambda x:  rolling_shifted(x, window))

    feat_cols = [f"win_pct_{window}g", f"gf_avg_{window}g", f"ga_avg_{window}g", f"gd_avg_{window}g"]

    # Merge back to home side
    home_feats = log[log["team"].isin(df["home_team"].unique())].copy()
    home_feats = home_feats[["date","team"] + feat_cols].rename(
        columns={"team": "home_team"} | {c: f"home_{c}" for c in feat_cols})

    away_feats = log[log["team"].isin(df["away_team"].unique())].copy()
    away_feats = away_feats[["date","team"] + feat_cols].rename(
        columns={"team": "away_team"} | {c: f"away_{c}" for c in feat_cols})

    # Take last record per team per date (avoid duplication from home+away)
    home_feats = home_feats.sort_values("date").groupby(["home_team","date"]).last().reset_index()
    away_feats = away_feats.sort_values("date").groupby(["away_team","date"]).last().reset_index()

    df = df.merge(home_feats, on=["date","home_team"], how="left")
    df = df.merge(away_feats, on=["date","away_team"], how="left")
    return df


# ─────────────────────────────────────────────────────────────
# Full feature build
# ─────────────────────────────────────────────────────────────

def build_matchup_features(force_refresh: bool = False) -> pd.DataFrame:
    """
    Master function: fetch → ELO → rolling form → matchup DataFrame.
    Target column: result_class (0=away win, 1=draw, 2=home win)
    """
    matches = fetch_all_matches(force_refresh=force_refresh)

    print("  Computing ELO ratings...")
    matches_elo, final_elo = compute_elo(matches)

    # Save final ELO snapshot for live predictions
    elo_path = SOCCER_RAW / "current_elo.json"
    elo_path.write_text(json.dumps(
        {k: round(v, 1) for k, v in sorted(final_elo.items(), key=lambda x: -x[1])},
        indent=2
    ))
    print(f"  Saved ELO for {len(final_elo)} teams to {elo_path.name}")

    print("  Adding rolling form features...")
    df = add_rolling_form(matches_elo, window=10)

    # Build differential features
    df["elo_diff"]       = df["home_elo_pre"] - df["away_elo_pre"]
    df["gd_diff_10g"]    = df.get("home_gd_avg_10g", 0) - df.get("away_gd_avg_10g", 0)
    df["win_pct_diff_10g"] = df.get("home_win_pct_10g", 0) - df.get("away_win_pct_10g", 0)

    # Target: 0=away win, 1=draw, 2=home win
    df["result_class"] = df["result"].map({-1: 0, 0: 1, 1: 2})

    cache = SOCCER_RAW / "matchups.csv"
    df.to_csv(cache, index=False)
    print(f"  Built {len(df)} international matchup rows")
    print(f"  Class distribution: {df['result_class'].value_counts().sort_index().to_dict()}")
    return df


def get_current_team_form(teams: list) -> dict:
    """
    Get rolling form for a list of team names from the cached match data.
    Returns {team_name: {gf_avg, ga_avg, win_pct, gd_avg}}.
    Used for live World Cup predictions.
    """
    cache = SOCCER_RAW / "all_matches.csv"
    if not cache.exists():
        return {}
    matches = pd.read_csv(cache, parse_dates=["date"])

    # Build per-team log (same logic as add_rolling_form)
    home_log = matches[["date","home_team","home_score","away_score","result"]].copy()
    home_log.columns = ["date","team","gf","ga","result_raw"]
    home_log["win"] = (home_log["result_raw"] == 1).astype(int)

    away_log = matches[["date","away_team","away_score","home_score","result"]].copy()
    away_log.columns = ["date","team","gf","ga","result_raw"]
    away_log["win"] = (away_log["result_raw"] == -1).astype(int)

    log = pd.concat([home_log, away_log]).sort_values(["team","date"]).reset_index(drop=True)
    log["gd"] = log["gf"] - log["ga"]

    result = {}
    for team in teams:
        tlog = log[log["team"] == team].tail(15)
        if len(tlog) < 3:
            continue
        result[team] = {
            "win_pct_10g": round(tlog["win"].mean(), 3),
            "gf_avg_10g":  round(tlog["gf"].mean(), 2),
            "ga_avg_10g":  round(tlog["ga"].mean(), 2),
            "gd_avg_10g":  round(tlog["gd"].mean(), 2),
        }
    return result


if __name__ == "__main__":
    df = build_matchup_features()
    print(df[["date","home_team","away_team","elo_diff","result_class"]].tail(5))
