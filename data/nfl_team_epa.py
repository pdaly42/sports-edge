"""
NFL team-level EPA per play, computed from nflfastR play-by-play.

EPA (Expected Points Added) per play is the single most predictive team-level
stat in modern NFL analytics — it captures scoring efficiency regardless of
pace or game script.

Features produced (per team, per week):
  off_epa_per_play       — offensive EPA/play on rush + pass
  def_epa_per_play       — EPA/play the team allows on defense
  off_pass_epa_per_play  — passing-game efficiency
  off_rush_epa_per_play  — rushing-game efficiency

Rolling 4-week and 8-week windows are added for each metric, shift(1) applied
so the current week's stats never leak into the current week's feature row.
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
import nfl_data_py as nfl

from config.settings import RAW_DIR

NFL_RAW = RAW_DIR / "nfl"
NFL_RAW.mkdir(parents=True, exist_ok=True)

METRIC_COLS = [
    "off_epa_per_play",
    "def_epa_per_play",
    "off_pass_epa_per_play",
    "off_rush_epa_per_play",
]


def fetch_team_epa_weekly(seasons: list) -> pd.DataFrame:
    """
    Aggregate play-by-play to team-week EPA metrics and add rolling windows.
    Cached to disk since the pbp download is ~200MB per season.
    """
    cache_path = NFL_RAW / f"team_epa_{min(seasons)}_{max(seasons)}.csv"
    if cache_path.exists():
        print(f"  Loading cached team EPA {min(seasons)}-{max(seasons)}")
        return pd.read_csv(cache_path)

    print(f"  Fetching NFL play-by-play {min(seasons)}-{max(seasons)} — this may take a minute...")
    cols = ["season", "week", "season_type", "posteam", "defteam",
            "epa", "pass", "rush"]
    pbp = nfl.import_pbp_data(seasons, columns=cols, downcast=True)
    pbp = pbp[pbp["season_type"] == "REG"].copy()
    pbp = pbp.dropna(subset=["posteam", "defteam", "epa"])
    # Only rush/pass plays — kicks, punts, penalties would dilute EPA
    pbp = pbp[(pbp["pass"] == 1) | (pbp["rush"] == 1)].copy()

    # Offensive aggregates keyed on possession team
    off_all = (
        pbp.groupby(["season", "week", "posteam"])
        .agg(off_epa_per_play=("epa", "mean"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    off_pass = (
        pbp[pbp["pass"] == 1]
        .groupby(["season", "week", "posteam"])
        .agg(off_pass_epa_per_play=("epa", "mean"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    off_rush = (
        pbp[pbp["rush"] == 1]
        .groupby(["season", "week", "posteam"])
        .agg(off_rush_epa_per_play=("epa", "mean"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    # Defensive aggregate keyed on the team on defense (their opponent's offense)
    def_all = (
        pbp.groupby(["season", "week", "defteam"])
        .agg(def_epa_per_play=("epa", "mean"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    merged = (
        off_all
        .merge(off_pass, on=["season", "week", "team"], how="left")
        .merge(off_rush, on=["season", "week", "team"], how="left")
        .merge(def_all,  on=["season", "week", "team"], how="left")
        .sort_values(["team", "season", "week"])
        .reset_index(drop=True)
    )

    for window in [4, 8]:
        for col in METRIC_COLS:
            merged[f"{col}_{window}w"] = (
                merged.groupby("team")[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=2).mean())
            )

    merged.to_csv(cache_path, index=False)
    print(f"  Saved {len(merged)} team-week EPA rows")
    return merged


def get_team_epa_for_matchups(seasons: list) -> pd.DataFrame:
    """DataFrame keyed on (season, week, team) with rolled EPA features."""
    epa = fetch_team_epa_weekly(seasons)
    roll_cols = [c for c in epa.columns if any(
        c.startswith(p + "_") for p in METRIC_COLS
    )]
    return epa[["season", "week", "team"] + roll_cols].copy()


def get_current_team_epa(seasons: list = None) -> pd.DataFrame:
    """Most recent rolling EPA per team — used for live predictions."""
    if seasons is None:
        current_year = pd.Timestamp.now().year
        seasons = list(range(current_year - 3, current_year + 1))
    # Bust cache so the current week reflects the latest game results
    cache_path = NFL_RAW / f"team_epa_{min(seasons)}_{max(seasons)}.csv"
    if cache_path.exists():
        cache_path.unlink()
    epa = fetch_team_epa_weekly(seasons)
    latest = epa.sort_values(["season", "week"]).groupby("team").last().reset_index()
    return latest


if __name__ == "__main__":
    df = get_team_epa_for_matchups(list(range(2020, pd.Timestamp.now().year + 1)))
    print(f"Team EPA feature rows: {len(df)}")
    print(df.head())
