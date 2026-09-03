"""
NFL team stats via ESPN's core API — live fallback for nflreadpy.

nflreadpy publishes team stats nightly after game days, which is fine for
midweek predictions but has two failure modes we want to insure against:

  1. Preseason / Week 1: the current-season parquet file may not exist yet
     (or be empty), so our schedule-derived rolling stats have nothing to
     work with for teams with no completed games.
  2. Sunday night → Monday morning: nflreadpy has a several-hour lag before
     that day's games appear in team_stats; ESPN has them within ~30 min.

This module reads ESPN's per-team seasonal statistics endpoint (public, no
API key). It returns a pandas DataFrame shaped like what
data.nfl_fetcher.get_current_team_stats() produces, so the fetcher can
merge it in without changing the model's feature expectations.

Only used as a fallback — nflreadpy remains the primary source since it
has richer, more consistent play-by-play/EPA data.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import requests
import pandas as pd
from datetime import datetime

from config.settings import RAW_DIR

ESPN_BASE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"

# ESPN's team_id → our 2-3 letter abbreviations used by nflreadpy/model
# (verified against espn.com; IDs 31/32 are unused by NFL).
ESPN_ID_TO_ABBR: dict[int, str] = {
    1:  "ATL", 2:  "BUF", 3:  "CHI", 4:  "CIN", 5:  "CLE",
    6:  "DAL", 7:  "DEN", 8:  "DET", 9:  "GB",  10: "TEN",
    11: "IND", 12: "KC",  13: "LV",  14: "LA",  15: "MIA",
    16: "MIN", 17: "NE",  18: "NO",  19: "NYG", 20: "NYJ",
    21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC", 25: "SF",
    26: "SEA", 27: "TB",  28: "WAS", 29: "CAR", 30: "JAX",
    33: "BAL", 34: "HOU",
}
ABBR_TO_ESPN_ID: dict[str, int] = {v: k for k, v in ESPN_ID_TO_ABBR.items()}


def _extract_stat(categories: list, cat_name: str, stat_name: str) -> float | None:
    """Pull one stat by (category, name); return None if either missing."""
    for cat in categories:
        if cat.get("name") == cat_name:
            for s in cat.get("stats", []):
                if s.get("name") == stat_name:
                    v = s.get("value")
                    return float(v) if v is not None else None
    return None


def fetch_team_season_totals(season: int, espn_id: int,
                              season_type: int = 2, timeout: int = 15) -> dict | None:
    """
    Season-to-date team totals from ESPN.

    season_type: 1=preseason, 2=regular, 3=postseason. Default regular.

    Returns a dict shaped for merge with nflreadpy schedule-derived rolling
    stats. Fields intentionally overlap the ones our feature engineering
    already looks for (pts_for_avg, pts_against_avg, win_pct, days_rest).
    """
    url = f"{ESPN_BASE}/seasons/{season}/types/{season_type}/teams/{espn_id}/statistics"
    try:
        r = requests.get(url, params={"limit": 1}, timeout=timeout)
        if not r.ok:
            return None
        data = r.json()
    except Exception:
        return None

    cats = (data.get("splits") or {}).get("categories") or []
    if not cats:
        return None

    # Wins/losses live in the "general" category on some ESPN team endpoints,
    # but the /statistics endpoint doesn't always include them. Pull what we
    # can and let the caller merge with schedule-derived data for the rest.
    pts_for_pg     = _extract_stat(cats, "scoring",   "totalPointsPerGame")
    net_yards_pg   = _extract_stat(cats, "passing",   "netYardsPerGame")
    pass_yards_pg  = _extract_stat(cats, "passing",   "passingYardsPerGame")
    rush_yards_pg  = _extract_stat(cats, "rushing",   "rushingYardsPerGame")
    completion_pct = _extract_stat(cats, "passing",   "completionPct")
    sacks_allowed  = _extract_stat(cats, "passing",   "sacks")
    def_sacks      = _extract_stat(cats, "defensive", "sacks")
    turnovers      = _extract_stat(cats, "general",   "turnovers") \
                     or _extract_stat(cats, "miscellaneous", "totalTakeaways")
    fumbles_lost   = _extract_stat(cats, "general",   "fumblesLost")

    return {
        "team":              ESPN_ID_TO_ABBR[espn_id],
        "season":            season,
        "espn_pts_for_avg":  pts_for_pg,
        "espn_net_yards_pg": net_yards_pg,
        "espn_pass_yds_pg":  pass_yards_pg,
        "espn_rush_yds_pg":  rush_yards_pg,
        "espn_comp_pct":     completion_pct,
        "espn_sacks_allowed": sacks_allowed,
        "espn_def_sacks":    def_sacks,
        "espn_turnovers":    turnovers,
        "espn_fumbles_lost": fumbles_lost,
    }


def fetch_all_team_totals(season: int, season_type: int = 2) -> pd.DataFrame:
    """One row per team for the given season/type. Silently drops teams
    the ESPN endpoint doesn't have data for."""
    rows = []
    for espn_id in ESPN_ID_TO_ABBR:
        d = fetch_team_season_totals(season, espn_id, season_type=season_type)
        if d and d.get("espn_pts_for_avg") is not None:
            rows.append(d)
    return pd.DataFrame(rows)


def get_current_espn_team_stats(current_year: int | None = None) -> pd.DataFrame:
    """
    Live ESPN team stats for use as a fallback.

    Tries the CURRENT regular season first. If ESPN's regular-season endpoint
    is empty (typical during preseason before any regular game has been
    played), falls back to the PRIOR regular season's ending totals. That way
    the caller always gets *some* team-level signal to blend in when
    nflreadpy has nothing for the current year yet.
    """
    if current_year is None:
        # NFL season year = year the regular season began (Sept-Feb spans two
        # calendar years). Aug or later → this-year is the current season.
        now = datetime.utcnow()
        current_year = now.year if now.month >= 8 else now.year - 1

    print(f"  ESPN fallback: fetching {current_year} regular-season team totals...")
    df = fetch_all_team_totals(current_year, season_type=2)
    if not df.empty:
        print(f"  ESPN fallback: got {len(df)} teams for {current_year} regular season")
        return df

    print(f"  ESPN fallback: {current_year} regular season empty; using {current_year - 1}")
    df = fetch_all_team_totals(current_year - 1, season_type=2)
    print(f"  ESPN fallback: got {len(df)} teams for {current_year - 1} regular season")
    return df


if __name__ == "__main__":
    df = get_current_espn_team_stats()
    print(f"\nColumns: {list(df.columns)}")
    print(df.head())
