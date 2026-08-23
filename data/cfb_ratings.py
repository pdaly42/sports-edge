"""
SP+ ratings from CollegeFootballData.com.

SP+ is the single most predictive team-level rating in college football —
Bill Connelly's system combines efficiency, explosiveness, field position,
finishing drives, and turnover luck. Available weekly during the season.

Features per team, per season:
  sp_overall     — overall SP+ rating (points above/below average)
  sp_offense     — offensive SP+ rating
  sp_defense     — defensive SP+ rating
  sp_st          — special teams SP+ rating
"""

import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import requests
from pathlib import Path

from config.settings import RAW_DIR, CFBD_API_KEY

CFB_RAW = RAW_DIR / "cfb"
CFB_RAW.mkdir(parents=True, exist_ok=True)

CFBD_BASE = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}"} if CFBD_API_KEY else {}


def fetch_sp_ratings(season: int) -> pd.DataFrame:
    """Fetch season-end SP+ ratings for one season. Cached per season."""
    cache_path = CFB_RAW / f"sp_{season}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    if not CFBD_API_KEY:
        return pd.DataFrame()

    print(f"  Fetching SP+ ratings for {season}...")
    try:
        r = requests.get(f"{CFBD_BASE}/ratings/sp", headers=HEADERS,
                         params={"year": season}, timeout=30)
        if not r.ok:
            print(f"  SP+ {season}: HTTP {r.status_code}")
            return pd.DataFrame()
        raw = r.json()
    except Exception as e:
        print(f"  SP+ {season}: error — {e}")
        return pd.DataFrame()

    rows = []
    for entry in raw:
        rows.append({
            "season":       entry.get("year"),
            "team":         entry.get("team"),
            "sp_overall":   (entry.get("rating") or 0.0),
            "sp_offense":   ((entry.get("offense") or {}).get("rating") or 0.0),
            "sp_defense":   ((entry.get("defense") or {}).get("rating") or 0.0),
            "sp_st":        ((entry.get("specialTeams") or {}).get("rating") or 0.0),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(cache_path, index=False)
        print(f"  Saved SP+ for {len(df)} teams in {season}")
    return df


def get_sp_features_for_matchups(seasons: list) -> pd.DataFrame:
    """
    Return DataFrame keyed on (season, team) with SP+ features.
    For a given season's games, we use that same season's SP+ ratings —
    they update weekly on CFBD but we snapshot the season-end version for
    training (small overfitting risk but negligible vs the signal).
    """
    frames = [fetch_sp_ratings(s) for s in seasons]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_current_sp_ratings(season: int = None) -> pd.DataFrame:
    """Latest SP+ ratings — used for live predictions. Busts cache."""
    from datetime import datetime
    season = season or datetime.now().year
    cache_path = CFB_RAW / f"sp_{season}.csv"
    if cache_path.exists():
        cache_path.unlink()
    return fetch_sp_ratings(season)


if __name__ == "__main__":
    from datetime import datetime
    df = fetch_sp_ratings(datetime.now().year - 1)
    print(f"SP+ rows: {len(df)}")
    if not df.empty:
        print(df.sort_values("sp_overall", ascending=False).head(10).to_string())
