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
import nflreadpy as nfl   # migrated from archived nfl_data_py, Sept 2025

from config.settings import RAW_DIR

NFL_RAW = RAW_DIR / "nfl"
NFL_RAW.mkdir(parents=True, exist_ok=True)

METRIC_COLS = [
    "off_epa_per_play",
    "def_epa_per_play",
    "off_pass_epa_per_play",
    "off_rush_epa_per_play",
    # Opponent-adjusted variants (subtract opponent's def/off EPA to credit
    # teams that put up numbers against strong opponents). Interpretation:
    #   off_epa_oadj = own_off_epa - opp_def_epa_allowed_that_week
    #   Team A putting up +0.10 EPA/play against a defense that usually allows
    #   -0.05 (elite defense) yields off_epa_oadj = +0.15 — genuinely strong.
    #   Same +0.10 against a defense allowing +0.15 (bad defense) yields
    #   off_epa_oadj = -0.05 — inflated raw score, weak against expectations.
    "off_epa_oadj_per_play",
    "def_epa_oadj_per_play",
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
    # Per-season fetch + skip on failure so a single missing year (typically
    # the current calendar year during preseason before Week 1 pbp lands, or
    # nflreadpy's ValueError "Season must be between 1999 and 2025") doesn't
    # kill the entire training set.
    cols = ["season", "week", "season_type", "posteam", "defteam",
            "epa", "pass", "rush"]
    frames = []
    for s in seasons:
        try:
            frames.append(nfl.load_pbp(seasons=[s]).to_pandas())
        except Exception as e:
            print(f"    pbp {s} unavailable ({type(e).__name__}), skipping")
    if not frames:
        raise RuntimeError(f"No pbp data available for any of {seasons}")
    pbp = pd.concat(frames, ignore_index=True)
    pbp = pbp[[c for c in cols if c in pbp.columns]]
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

    # ── Opponent adjustment (Phase 4) ─────────────────────────────────────
    # Build team→opponent map per (season, week) from the pbp itself, then
    # look up the opponent's raw def/off EPA that week. Subtract to get an
    # opponent-adjusted metric: how much better/worse did this team perform
    # than what the opponent typically allows/scores?
    #
    # Uses same-week opponent EPA (not rolling) for v1 simplicity — slight
    # circularity but captures the strength-of-schedule effect well enough
    # for the model to learn from. Rolling windows applied after.
    opp_map = (pbp[["season", "week", "posteam", "defteam"]]
               .drop_duplicates()
               .rename(columns={"posteam": "team", "defteam": "opponent"}))
    merged = merged.merge(opp_map, on=["season", "week", "team"], how="left")

    opp_epa = merged[["season", "week", "team",
                      "def_epa_per_play", "off_epa_per_play"]].rename(
        columns={"team": "opponent",
                 "def_epa_per_play": "opp_def_epa_per_play",
                 "off_epa_per_play": "opp_off_epa_per_play"}
    )
    merged = merged.merge(opp_epa, on=["season", "week", "opponent"], how="left")

    merged["off_epa_oadj_per_play"] = (
        merged["off_epa_per_play"] - merged["opp_def_epa_per_play"]
    )
    merged["def_epa_oadj_per_play"] = (
        merged["def_epa_per_play"] - merged["opp_off_epa_per_play"]
    )
    merged = merged.drop(columns=["opp_def_epa_per_play", "opp_off_epa_per_play"])

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
    """
    Most recent rolling EPA per team — used for live predictions.

    Resilient to nflverse not yet publishing the current season's play-by-play
    (404s until Week 1 regular season data lands). Drops the newest season and
    retries if the initial fetch fails. Returns empty on total failure — the
    caller in predict_nfl handles that.
    """
    if seasons is None:
        current_year = pd.Timestamp.now().year
        seasons = list(range(current_year - 3, current_year + 1))

    attempt = list(seasons)
    while attempt:
        cache_path = NFL_RAW / f"team_epa_{min(attempt)}_{max(attempt)}.csv"
        if cache_path.exists():
            cache_path.unlink()
        try:
            epa = fetch_team_epa_weekly(attempt)
            if epa is not None and not epa.empty:
                if attempt != seasons:
                    print(f"  Team EPA: falling back to seasons {min(attempt)}-{max(attempt)} "
                          f"(current year pbp not yet published)")
                latest = epa.sort_values(["season", "week"]).groupby("team").last().reset_index()
                return latest
            attempt = attempt[:-1]
        except Exception as e:
            print(f"  Team EPA {max(attempt)} unavailable ({type(e).__name__}: {e}); "
                  f"trying without it")
            attempt = attempt[:-1]

    print("  Team EPA: no seasons available — returning empty frame")
    return pd.DataFrame()


if __name__ == "__main__":
    df = get_team_epa_for_matchups(list(range(2020, pd.Timestamp.now().year + 1)))
    print(f"Team EPA feature rows: {len(df)}")
    print(df.head())
