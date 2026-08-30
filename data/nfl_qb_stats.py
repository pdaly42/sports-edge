"""
NFL QB stats via nflreadpy (successor to archived nfl_data_py).

Identifies the starter each week (QB with most attempts) and computes
rolling performance metrics used as features in the prediction model.

Features produced (per team, per week):
  qb_epa_per_att   — passing EPA per attempt (best single predictor)
  qb_comp_pct      — completion percentage
  qb_ypa           — yards per attempt
  qb_td_int_ratio  — (TDs+1)/(INTs+1), Laplace-smoothed

Rolling windows of 4 and 8 weeks are added for each metric.
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

# Minimum attempts to count a QB as the starter that week
MIN_ATTEMPTS = 10


def _compute_qb_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-game QB efficiency metrics to a weekly QB row."""
    df = df.copy()
    df["qb_epa_per_att"]  = df["passing_epa"] / df["attempts"].replace(0, np.nan)
    df["qb_comp_pct"]     = df["completions"] / df["attempts"].replace(0, np.nan)
    df["qb_ypa"]          = df["passing_yards"] / df["attempts"].replace(0, np.nan)
    df["qb_td_int_ratio"] = (df["passing_tds"] + 1) / (df["interceptions"] + 1)
    return df


def fetch_qb_weekly(seasons: list) -> pd.DataFrame:
    """
    Download weekly QB stats for all seasons, identify the starter per team/week
    (highest attempts), and return one row per (season, week, team).
    Caches to disk.
    """
    cache_path = NFL_RAW / f"qb_weekly_{min(seasons)}_{max(seasons)}.csv"
    if cache_path.exists():
        print(f"  Loading cached QB stats {min(seasons)}-{max(seasons)}")
        return pd.read_csv(cache_path)

    print(f"  Fetching QB weekly stats {min(seasons)}-{max(seasons)}...")
    # nflreadpy's player_stats replaces nfl_data_py's weekly_data. Same shape
    # (one row per player per week) but under a new nflverse-data release
    # path. Fetch per-season and skip any year that 404s (typically the
    # current calendar year during preseason before Week 1 data lands) so
    # a single missing year doesn't kill the whole training pipeline.
    per_season = []
    for s in seasons:
        try:
            per_season.append(nfl.load_player_stats(seasons=[s], summary_level="week").to_pandas())
        except Exception as e:
            print(f"    QB weekly {s} unavailable ({type(e).__name__}), skipping")
    if not per_season:
        raise RuntimeError(f"No QB weekly data available for any of {seasons}")
    raw = pd.concat(per_season, ignore_index=True)
    raw = raw[raw["season_type"] == "REG"].copy()

    qbs = raw[
        (raw["position"] == "QB") & (raw["attempts"] >= MIN_ATTEMPTS)
    ][["player_name", "recent_team", "season", "week",
       "completions", "attempts", "passing_yards",
       "passing_tds", "interceptions", "passing_epa"]].copy()
    qbs = qbs.rename(columns={"recent_team": "team"})

    # One starter row per team/week — highest attempts
    qbs = qbs.sort_values("attempts", ascending=False)
    qbs = qbs.drop_duplicates(subset=["team", "season", "week"], keep="first")

    qbs = _compute_qb_metrics(qbs)
    qbs = qbs.sort_values(["team", "season", "week"]).reset_index(drop=True)

    METRIC_COLS = ["qb_epa_per_att", "qb_comp_pct", "qb_ypa", "qb_td_int_ratio"]
    for window in [4, 8]:
        for col in METRIC_COLS:
            qbs[f"{col}_{window}w"] = (
                qbs.groupby("team")[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=2).mean())
            )

    qbs.to_csv(cache_path, index=False)
    print(f"  Saved {len(qbs)} QB starter rows")
    return qbs


def get_qb_features_for_matchups(seasons: list) -> pd.DataFrame:
    """
    Return a DataFrame keyed on (season, week, team) with rolled QB features
    ready to join onto the matchup DataFrame.
    """
    qbs = fetch_qb_weekly(seasons)
    roll_cols = [c for c in qbs.columns if any(
        c.startswith(p) for p in ["qb_epa_per_att_", "qb_comp_pct_", "qb_ypa_", "qb_td_int_ratio_"]
    )]
    return qbs[["season", "week", "team"] + roll_cols].copy()


def get_current_qb_stats(seasons: list = None) -> pd.DataFrame:
    """
    Most recent week's starter QB rolling stats per team — used for live predictions.
    Clears the cache so we always pull the freshest data.

    Resilient to nflverse not yet publishing the current season's weekly data
    (which 404s until roughly after Week 1 of the regular season). If the full
    range fails, progressively drops the most recent season and retries. If
    every season fails, returns an empty DataFrame — predict_nfl handles that
    gracefully by treating QB features as neutral (0.0).
    """
    if seasons is None:
        current_year = pd.Timestamp.now().year
        seasons = list(range(current_year - 3, current_year + 1))

    attempt = list(seasons)
    while attempt:
        # Bust cache for this exact range so the current week reflects reality
        cache_path = NFL_RAW / f"qb_weekly_{min(attempt)}_{max(attempt)}.csv"
        if cache_path.exists():
            cache_path.unlink()
        try:
            qbs = fetch_qb_weekly(attempt)
            if qbs is not None and not qbs.empty:
                if attempt != seasons:
                    print(f"  QB stats: falling back to seasons {min(attempt)}-{max(attempt)} "
                          f"(current year not yet published)")
                latest = qbs.sort_values(["season", "week"]).groupby("team").last().reset_index()
                return latest
            # Empty result — treat like a soft 404
            attempt = attempt[:-1]
        except Exception as e:
            print(f"  QB weekly {max(attempt)} unavailable ({type(e).__name__}: {e}); "
                  f"trying without it")
            attempt = attempt[:-1]

    print("  QB weekly stats: no seasons available — returning empty frame")
    return pd.DataFrame()


if __name__ == "__main__":
    df = get_qb_features_for_matchups(list(range(2018, 2025)))
    print(f"QB feature rows: {len(df)}")
    print(df.head())
