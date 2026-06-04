"""
MLB pitcher stats utilities.

Two jobs:
  1. build_season_pitcher_lookup(seasons) → {season: {last_name: {era, whip, ...}}}
     Used to enrich historical training data.

  2. get_probable_pitcher_stats(date_str) → {team_name: {era, whip, ...}}
     Uses the free MLB Stats API to get today's probable starters + their
     current season stats. No API key required.
"""

import sys, os, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from config.settings import RAW_DIR

PITCHER_CACHE = RAW_DIR / "mlb" / "pitcher_stats"
PITCHER_CACHE.mkdir(parents=True, exist_ok=True)

MLB_STATS_API = "https://statsapi.mlb.com/api/v1"

# Stats we care about — all available from both BRef and MLB Stats API
PITCHER_FEATURE_COLS = ["era", "whip", "k_per_9", "k_bb_ratio", "innings_pitched"]


# ─────────────────────────────────────────────────────────────
# Historical: BRef season stats → per-pitcher lookup
# ─────────────────────────────────────────────────────────────

def _fetch_bref_season(season: int) -> pd.DataFrame:
    cache = PITCHER_CACHE / f"bref_{season}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    print(f"    Fetching BRef pitcher stats {season}...")
    time.sleep(2)
    from pybaseball import pitching_stats_bref
    df = pitching_stats_bref(season)
    df.to_csv(cache, index=False)
    return df


def build_season_pitcher_lookup(seasons: list) -> dict:
    """
    Returns a nested dict:
      {season: {(last_name, team_abbr): {era, whip, k_per_9, k_bb_ratio, ip}}}

    Keyed by (last_name, team) so we can disambiguate pitchers with identical
    last names who play for different teams.
    Falls back to last_name alone when team is unknown.
    """
    lookup = {}
    for season in seasons:
        df = _fetch_bref_season(season)

        # Only keep pitchers with at least 5 starts (filters out pure relievers)
        starters = df[df["GS"] >= 5].copy()

        season_dict = {}
        for _, row in starters.iterrows():
            full_name = str(row["Name"]).strip()
            last_name = full_name.split()[-1].lower()
            team      = str(row["Tm"]).strip().upper()

            # Handle "TOT" (traded mid-season — multiple team rows, take the TOT row)
            # TOT has combined season stats — use those and map to both teams
            stats = {
                "era":             _safe(row.get("ERA")),
                "whip":            _safe(row.get("WHIP")),
                "k_per_9":         _safe(row.get("SO9")),
                "k_bb_ratio":      _safe(row.get("SO/W")),
                "innings_pitched": _safe(row.get("IP")),
            }

            # Store by (last_name, team)
            season_dict[(last_name, team)] = stats
            # Also store by last_name alone as fallback
            if last_name not in season_dict:
                season_dict[last_name] = stats
            elif team == "TOT":
                season_dict[last_name] = stats  # prefer combined row

        lookup[season] = season_dict
    return lookup


def _safe(val):
    try:
        f = float(val)
        return round(f, 3) if not np.isnan(f) else None
    except (TypeError, ValueError):
        return None


def lookup_pitcher(season_dict: dict, last_name: str, team: Optional[str] = None):
    """
    Look up a pitcher in a season dict.
    Returns stat dict or None if not found.
    """
    last = last_name.lower().strip()
    if team:
        result = season_dict.get((last, team.upper()))
        if result:
            return result
    # Fallback to last name only
    return season_dict.get(last)


# ─────────────────────────────────────────────────────────────
# Live: MLB Stats API probable pitchers for a given date
# ─────────────────────────────────────────────────────────────

def get_probable_pitcher_stats(date_str: str) -> dict:
    """
    Fetch today's probable starters via the free MLB Stats API.
    Returns {odds_api_team_name: pitcher_stat_dict}.

    pitcher_stat_dict keys: era, whip, k_per_9, k_bb_ratio, innings_pitched, name
    """
    # Step 1: Get schedule with probable pitchers
    url = f"{MLB_STATS_API}/schedule"
    params = {
        "sportId": 1,
        "date":    date_str,
        "hydrate": "probablePitcher,team",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  MLB Stats API error: {e}")
        return {}

    dates = data.get("dates", [])
    if not dates:
        return {}

    result = {}
    pitcher_ids = {}  # team_name → pitcher_id

    for game in dates[0].get("games", []):
        for side in ("home", "away"):
            team_data = game["teams"][side]
            team_name = team_data["team"]["name"]
            pitcher   = team_data.get("probablePitcher")
            if pitcher:
                pitcher_ids[team_name] = (pitcher["id"], pitcher.get("fullName", ""))

    if not pitcher_ids:
        print("  No probable pitchers posted yet for this date")
        return {}

    # Step 2: Fetch current season stats for each pitcher
    current_year = int(date_str[:4])
    for team_name, (pid, pname) in pitcher_ids.items():
        stats = _fetch_pitcher_season_stats(pid, current_year)
        if stats:
            stats["name"] = pname
            result[team_name] = stats
            print(f"    {team_name}: {pname} — ERA {stats.get('era')} WHIP {stats.get('whip')}")
        else:
            print(f"    {team_name}: {pname} — no stats yet (new/early season)")

    return result


def _fetch_pitcher_season_stats(pitcher_id: int, season: int) -> Optional[dict]:
    """Fetch one pitcher's season stats from MLB Stats API."""
    url = f"{MLB_STATS_API}/people/{pitcher_id}/stats"
    params = {"stats": "season", "group": "pitching", "season": season}
    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        splits = resp.json().get("stats", [{}])[0].get("splits", [])
        if not splits:
            return None
        s = splits[0]["stat"]
        ip  = _parse_ip(s.get("inningsPitched", "0"))
        so  = float(s.get("strikeOuts", 0))
        bb  = float(s.get("baseOnBalls", 0))
        return {
            "era":             _safe(s.get("era")),
            "whip":            _safe(s.get("whip")),
            "k_per_9":         round(so / ip * 9, 2) if ip > 0 else None,
            "k_bb_ratio":      round(so / bb, 2) if bb > 0 else None,
            "innings_pitched": round(ip, 1),
        }
    except Exception:
        return None


def _parse_ip(ip_str: str) -> float:
    """Convert '123.2' innings format to decimal (123.2 = 123 + 2/3 innings)."""
    try:
        parts = str(ip_str).split(".")
        full  = int(parts[0])
        frac  = int(parts[1]) / 3 if len(parts) > 1 else 0
        return full + frac
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────
# Median fallback — used when a starter can't be matched
# ─────────────────────────────────────────────────────────────

LEAGUE_MEDIANS = {
    "era":             4.25,
    "whip":            1.30,
    "k_per_9":         8.5,
    "k_bb_ratio":      2.5,
    "innings_pitched": 130.0,
}


def pitcher_stats_or_median(stats: Optional[dict]) -> dict:
    """Return stats dict, filling any missing values with league medians."""
    if stats is None:
        return LEAGUE_MEDIANS.copy()
    return {k: (stats.get(k) if stats.get(k) is not None else LEAGUE_MEDIANS[k])
            for k in LEAGUE_MEDIANS}
