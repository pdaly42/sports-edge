"""
NFL injury report processing.

Computes a weighted injury impact score per team per week from the official
NFL injury report. Scores are used as model features.

Design principles:
  - Position weight: how much does losing a player at this position hurt?
    QB is treated separately at a much higher weight because a starter QB
    being out shifts win probability by 10-15 percentage points on its own.
  - Status multiplier: how likely is the player to miss/be limited?
    Out=1.0, Doubtful=0.75, Questionable=0.35 (roughly the historical
    "Questionable" game-miss rate).
  - Depth discount: the 2nd injured player at a position counts for 60%,
    3rd for 36%, etc. (geometric decay). Losing two CBs hurts less than
    twice losing one CB.
  - QB is broken out as its own feature (qb_injury_impact) because the
    model should learn its weight independently — it's so much larger than
    other positions that mixing it into a single score distorts the scale.
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
import nflreadpy as nfl   # migrated from archived nfl_data_py, Sept 2025

from config.settings import RAW_DIR

NFL_RAW = RAW_DIR / "nfl"
NFL_RAW.mkdir(parents=True, exist_ok=True)

# How much does losing the *first* starter at each position hurt the team?
# Scale: 1.0 = as impactful as losing an elite starter; 0.1 = barely matters.
POSITION_BASE_WEIGHT: dict[str, float] = {
    "QB": 1.00,   # starter out → massive win-prob swing
    "T":  0.38,   # offensive tackle (blindside especially)
    "G":  0.22,
    "C":  0.20,
    "WR": 0.38,   # WR1 caliber
    "TE": 0.28,
    "RB": 0.32,
    "FB": 0.12,
    "CB": 0.30,
    "S":  0.24,
    "LB": 0.26,
    "DE": 0.28,
    "DT": 0.20,
    "K":  0.14,
    "P":  0.08,
    "LS": 0.05,
}
DEFAULT_POSITION_WEIGHT = 0.15   # for any position not listed above

# Probability the player materially misses game time / plays at reduced capacity
STATUS_MULTIPLIER: dict[str, float] = {
    "Out":          1.00,
    "Doubtful":     0.75,
    "Questionable": 0.35,
    "Probable":     0.05,   # rarely appears but include for completeness
}

# Geometric depth discount: 2nd player at same position = 60%, 3rd = 36%, …
DEPTH_DECAY = 0.60


def _team_injury_score(group: pd.DataFrame) -> dict:
    """
    Given all injured players for one (team, week), return:
      injury_score    — aggregate non-QB impact
      qb_injury_impact — QB-specific impact (0 if QB healthy)
    """
    # Sort by position weight descending so depth discount applies correctly
    group = group.copy()
    group["_pos_w"] = group["position"].map(POSITION_BASE_WEIGHT).fillna(DEFAULT_POSITION_WEIGHT)
    group["_stat_m"] = group["report_status"].map(STATUS_MULTIPLIER).fillna(0.0)
    group["_raw"] = group["_pos_w"] * group["_stat_m"]

    # Separate QB from everyone else
    qb_rows    = group[group["position"] == "QB"].sort_values("_raw", ascending=False)
    other_rows = group[group["position"] != "QB"].sort_values(["position", "_raw"], ascending=[True, False])

    # QB impact: only starter matters; backup QB being out barely moves the needle
    # We take the highest-impact QB row (starter most likely to miss), then discount
    # any additional QBs at DEPTH_DECAY^1, DEPTH_DECAY^2, …
    qb_impact = sum(
        row["_raw"] * (DEPTH_DECAY ** i)
        for i, (_, row) in enumerate(qb_rows.iterrows())
    )

    # Non-QB injury score: within each position, apply depth discount
    non_qb_score = 0.0
    for pos, pos_group in other_rows.groupby("position"):
        for i, (_, row) in enumerate(pos_group.iterrows()):
            non_qb_score += row["_raw"] * (DEPTH_DECAY ** i)

    return {"qb_injury_impact": round(qb_impact, 4),
            "injury_score":     round(non_qb_score, 4)}


def fetch_injury_reports(seasons: list) -> pd.DataFrame:
    """Download and cache NFL injury reports for the given seasons."""
    cache_path = NFL_RAW / f"injuries_{min(seasons)}_{max(seasons)}.csv"
    if cache_path.exists():
        print(f"  Loading cached injury reports {min(seasons)}-{max(seasons)}")
        return pd.read_csv(cache_path)

    print(f"  Fetching injury reports {min(seasons)}-{max(seasons)}...")
    # Per-season fetch + skip on failure — nflreadpy raises "Season must be
    # between 2009 and 2025" for 2026 (during preseason). Individual retries
    # keep the training pipeline moving with whatever data is available.
    frames = []
    for s in seasons:
        try:
            frames.append(nfl.load_injuries(seasons=[s]).to_pandas())
        except Exception as e:
            print(f"    injuries {s} unavailable ({type(e).__name__}), skipping")
    if not frames:
        raise RuntimeError(f"No injury data available for any of {seasons}")
    raw = pd.concat(frames, ignore_index=True)
    cols = ["season", "week", "team", "full_name", "position", "report_status"]
    df = raw[cols].dropna(subset=["report_status"]).copy()
    # Keep only statuses we have multipliers for
    df = df[df["report_status"].isin(STATUS_MULTIPLIER)].copy()
    df.to_csv(cache_path, index=False)
    print(f"  Saved {len(df)} injury report rows")
    return df


def compute_injury_features(seasons: list) -> pd.DataFrame:
    """
    Return a DataFrame with columns (season, week, team, injury_score, qb_injury_impact)
    for every (team, week) that appears in the injury report.
    Zero-filled for teams not on the report that week.
    """
    raw = fetch_injury_reports(seasons)

    rows = []
    for (season, week, team), group in raw.groupby(["season", "week", "team"]):
        scores = _team_injury_score(group)
        rows.append({"season": season, "week": week, "team": team, **scores})

    return pd.DataFrame(rows)


def get_injury_features_for_matchups(seasons: list) -> pd.DataFrame:
    """Wrapper that returns injury features keyed on (season, week, team)."""
    return compute_injury_features(seasons)


def get_current_injury_scores(season: int, week: int) -> dict[str, dict]:
    """
    Fetch the injury report for the given season/week and return a dict:
      { team_abbr: {"injury_score": float, "qb_injury_impact": float} }

    Busts the cache so we always get the latest report during the week.
    """
    cache_path = NFL_RAW / f"injuries_{season}_{season}.csv"
    if cache_path.exists():
        cache_path.unlink()

    raw = fetch_injury_reports([season])
    this_week = raw[(raw["season"] == season) & (raw["week"] == week)]
    if this_week.empty:
        print(f"  No injury data for {season} week {week}")
        return {}

    result = {}
    for (_, _, team), group in this_week.groupby(["season", "week", "team"]):
        result[team] = _team_injury_score(group)
    return result


if __name__ == "__main__":
    df = compute_injury_features([2022, 2023])
    print(f"Injury feature rows: {len(df)}")
    print(df.sort_values("qb_injury_impact", ascending=False).head(10).to_string())
