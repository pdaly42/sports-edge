"""
CFB team-level EPA per play from CollegeFootballData.com.

Note: CFBD's /plays endpoint is heavy — one season is 15-25MB of JSON with
several hundred thousand plays. We cache per season and use the season-week
pre-aggregated /stats/season/advanced endpoint where possible.

Approach:
  - Use /ppa/games endpoint for per-game offensive/defensive PPA (predicted
    points added, CFBD's equivalent of EPA).
  - Aggregate to (season, week, team) with rolling 3-week and 6-week windows.
    CFB seasons are 12-15 games so shorter windows than NFL.

Features per team, per week:
  cfb_off_epa_per_play, cfb_def_epa_per_play
  cfb_off_pass_epa, cfb_off_rush_epa
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import requests
from pathlib import Path

from config.settings import RAW_DIR, CFBD_API_KEY

CFB_RAW = RAW_DIR / "cfb"
CFB_RAW.mkdir(parents=True, exist_ok=True)

CFBD_BASE = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}"} if CFBD_API_KEY else {}

METRIC_COLS = [
    "cfb_off_epa_per_play",
    "cfb_def_epa_per_play",
    "cfb_off_pass_epa",
    "cfb_off_rush_epa",
]


def fetch_ppa_games(season: int) -> pd.DataFrame:
    """
    Per-game PPA (predicted points added) for one season.
    CFBD's /ppa/games returns offensive PPA per team per game with pass/rush breakdowns.
    Cached per season.
    """
    cache_path = CFB_RAW / f"ppa_games_{season}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    if not CFBD_API_KEY:
        return pd.DataFrame()

    print(f"  Fetching CFB PPA per game for {season}...")
    try:
        r = requests.get(f"{CFBD_BASE}/ppa/games", headers=HEADERS,
                         params={"year": season, "seasonType": "regular"}, timeout=60)
        if not r.ok:
            print(f"  CFB PPA {season}: HTTP {r.status_code}")
            return pd.DataFrame()
        raw = r.json()
    except Exception as e:
        print(f"  CFB PPA {season}: error — {e}")
        return pd.DataFrame()

    rows = []
    for entry in raw:
        offense = entry.get("offense") or {}
        rows.append({
            "season":  entry.get("season"),
            "week":    entry.get("week"),
            "team":    entry.get("team"),
            "opponent": entry.get("opponent"),
            "cfb_off_epa_per_play": offense.get("overall") or 0.0,
            "cfb_off_pass_epa":     offense.get("passing") or 0.0,
            "cfb_off_rush_epa":     offense.get("rushing") or 0.0,
        })
    df = pd.DataFrame(rows)

    # Defensive PPA = opponent's offensive PPA when they were on offense
    # For each (season, week, defteam) join the opponent's offense number
    if not df.empty:
        opp = df[["season", "week", "team", "cfb_off_epa_per_play"]].rename(
            columns={"team": "opponent", "cfb_off_epa_per_play": "cfb_def_epa_per_play"}
        )
        df = df.merge(opp, on=["season", "week", "opponent"], how="left")
        df["cfb_def_epa_per_play"] = df["cfb_def_epa_per_play"].fillna(0.0)

    if not df.empty:
        df.to_csv(cache_path, index=False)
        print(f"  Saved {len(df)} team-game PPA rows for {season}")
    return df


def fetch_team_epa_weekly(seasons: list) -> pd.DataFrame:
    """
    Aggregate per-game PPA into rolling per-team-per-week features.
    3-game and 6-game rolling windows with shift(1) for leakage safety.
    """
    frames = [fetch_ppa_games(s) for s in seasons]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    df = df.sort_values(["team", "season", "week"]).reset_index(drop=True)

    for window in [3, 6]:
        for col in METRIC_COLS:
            df[f"{col}_{window}g"] = (
                df.groupby("team")[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=2).mean())
            )
    return df


def get_team_epa_for_matchups(seasons: list) -> pd.DataFrame:
    """DataFrame keyed on (season, week, team) with rolled EPA features."""
    epa = fetch_team_epa_weekly(seasons)
    if epa.empty:
        return epa
    roll_cols = [c for c in epa.columns if any(
        c.startswith(p + "_") for p in METRIC_COLS
    )]
    return epa[["season", "week", "team"] + roll_cols].copy()


def get_current_team_epa(seasons: list = None) -> pd.DataFrame:
    """Latest rolling EPA per team — for live predictions."""
    from datetime import datetime
    if seasons is None:
        current_year = datetime.now().year
        seasons = list(range(current_year - 2, current_year + 1))
    # Bust current season cache
    p = CFB_RAW / f"ppa_games_{datetime.now().year}.csv"
    if p.exists():
        p.unlink()
    epa = fetch_team_epa_weekly(seasons)
    if epa.empty:
        return epa
    return epa.sort_values(["season", "week"]).groupby("team").last().reset_index()


if __name__ == "__main__":
    from datetime import datetime
    df = get_team_epa_for_matchups([datetime.now().year - 1])
    print(f"CFB EPA rows: {len(df)}")
    if not df.empty:
        print(df.head())
