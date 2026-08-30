"""
Fetch NFL game results via nflreadpy (wraps nflfastR data, free, no API key).
Saves raw data to data/raw/nfl/ and returns engineered matchup DataFrames.

Game data columns used: season, game_id, gameday, home_team, away_team,
home_score, away_score, home_rest, away_rest, div_game, roof, surface.
"""

import pandas as pd
import numpy as np
import sys, os
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
import nflreadpy as nfl   # migrated from archived nfl_data_py, Sept 2025

from config.settings import RAW_DIR
from utils.features import rolling_avg, days_rest

NFL_RAW = RAW_DIR / "nfl"
NFL_RAW.mkdir(parents=True, exist_ok=True)

# Seasons with (possibly) complete results — auto-extends to current calendar
# year so a mid-2026 run automatically pulls the finished 2025 season plus any
# early 2026 games that have completed. Uncompleted games are filtered out by
# fetch_season_games via the home_score/away_score notna check.
DEFAULT_SEASONS = list(range(2010, datetime.now().year + 1))


def fetch_season_games(seasons: list) -> pd.DataFrame:
    """
    Fetch NFL game schedule/results for the given seasons from nfl_data_py.
    Filters to completed regular-season games only.
    Caches by season range to avoid re-downloading.
    """
    cache_path = NFL_RAW / f"games_{min(seasons)}_{max(seasons)}.csv"
    if cache_path.exists():
        print(f"  Loading cached NFL games {min(seasons)}-{max(seasons)}")
        df = pd.read_csv(cache_path, parse_dates=["gameday"])
        return df

    print(f"  Fetching NFL games {min(seasons)}-{max(seasons)} via nfl_data_py...")
    # nflreadpy returns polars; convert at the boundary so all downstream code
    # can keep using pandas operations. seasons= is the same list-of-int shape.
    raw = nfl.load_schedules(seasons=seasons).to_pandas()

    # Keep only completed regular-season games
    raw = raw[raw["game_type"] == "REG"].copy()
    raw = raw[raw["home_score"].notna() & raw["away_score"].notna()].copy()

    cols = [
        "season", "week", "game_id", "gameday", "home_team", "away_team",
        "home_score", "away_score",
        "home_rest", "away_rest",     # days rest provided by nfl_data_py
        "div_game",
        # Weather / venue — used by the totals model
        "roof", "surface", "temp", "wind",
    ]
    # Keep only available columns
    cols = [c for c in cols if c in raw.columns]
    df = raw[cols].copy()
    df["gameday"] = pd.to_datetime(df["gameday"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    df.to_csv(cache_path, index=False)
    print(f"  Saved {len(df)} games")
    return df


def build_team_game_log(seasons: list = None) -> pd.DataFrame:
    """
    Per-team-per-game log with rolling engineered features.
    Each game appears twice: home perspective and away perspective.
    """
    if seasons is None:
        seasons = DEFAULT_SEASONS

    raw = fetch_season_games(seasons)

    home = raw.copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["points_for"] = home["home_score"]
    home["points_against"] = home["away_score"]
    home["is_home"] = 1
    home["rest"] = home.get("home_rest", np.nan)

    away = raw.copy()
    away["team"] = away["away_team"]
    away["opponent"] = away["home_team"]
    away["points_for"] = away["away_score"]
    away["points_against"] = away["home_score"]
    away["is_home"] = 0
    away["rest"] = away.get("away_rest", np.nan)

    keep = ["gameday", "season", "week", "game_id", "team", "opponent", "is_home",
            "points_for", "points_against", "rest"]
    if "div_game" in raw.columns:
        home["div_game"] = raw["div_game"]
        away["div_game"] = raw["div_game"]
        keep.append("div_game")

    log = pd.concat([home[keep], away[keep]], ignore_index=True)
    log = log.rename(columns={"gameday": "date"})
    log["win"] = (log["points_for"] > log["points_against"]).astype(int)
    log["point_diff"] = log["points_for"] - log["points_against"]
    log = log.sort_values(["team", "date"]).reset_index(drop=True)

    # Rolling stats — shift(1) inside rolling_avg prevents leakage
    for window in [4, 8, 16]:
        log[f"win_pct_{window}g"]       = rolling_avg(log, "win",          window)
        log[f"pts_for_avg_{window}g"]   = rolling_avg(log, "points_for",   window)
        log[f"pts_against_avg_{window}g"] = rolling_avg(log, "points_against", window)
        log[f"point_diff_avg_{window}g"] = rolling_avg(log, "point_diff",  window)

    # Season-to-date win % and point diff (no leakage — shift(1) applied)
    log["season_win_pct"] = (
        log.groupby(["team", "season"])["win"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    log["season_point_diff_avg"] = (
        log.groupby(["team", "season"])["point_diff"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    return log


def build_matchup_features(seasons: list = None, include_qb: bool = True,
                           include_injuries: bool = True) -> pd.DataFrame:
    """
    Per-game matchup DataFrame for model training.
    Target: home_win (1 = home team won).

    Optionally merges in:
      - QB rolling efficiency stats (qb_epa_per_att, comp_pct, ypa, td_int_ratio)
      - Position-weighted injury impact scores (injury_score, qb_injury_impact)

    QB and injury features default to 0 when unavailable (early seasons,
    missing injury report) so no rows are dropped.
    """
    if seasons is None:
        seasons = DEFAULT_SEASONS

    log = build_team_game_log(seasons)

    home_log = log[log["is_home"] == 1].copy()
    away_log = log[log["is_home"] == 0].copy()

    roll_cols = [c for c in log.columns if any(
        c.startswith(p) for p in
        ["win_pct_", "pts_for_avg_", "pts_against_avg_", "point_diff_avg_",
         "season_win_pct", "season_point_diff_avg"]
    )]

    # Bring raw scores + venue/weather into home_log so we can compute total_points
    # and engineer weather features on the matchup row.
    raw_full = fetch_season_games(seasons)
    weather_cols = [c for c in ["roof", "surface", "temp", "wind"] if c in raw_full.columns]
    raw_scores = raw_full[["gameday", "home_team", "away_team", "home_score", "away_score"] + weather_cols].copy()
    raw_scores = raw_scores.rename(columns={"gameday": "date"})
    raw_scores["date"] = pd.to_datetime(raw_scores["date"])

    home_feats = home_log[["date", "week", "team", "opponent", "win", "season", "rest"] + roll_cols].copy()
    home_feats.columns = (
        ["date", "week", "home_team", "away_team", "home_win", "season", "home_rest"]
        + [f"home_{c}" for c in roll_cols]
    )
    home_feats = home_feats.merge(raw_scores, on=["date", "home_team", "away_team"], how="left")
    home_feats["total_points"] = home_feats["home_score"] + home_feats["away_score"]

    away_feats = away_log[["date", "team", "rest"] + roll_cols].copy()
    away_feats.columns = (
        ["date", "away_team", "away_rest"]
        + [f"away_{c}" for c in roll_cols]
    )

    matchups = home_feats.merge(away_feats, on=["date", "away_team"])

    # Differential features
    for window in [4, 8, 16]:
        matchups[f"win_pct_diff_{window}g"] = (
            matchups[f"home_win_pct_{window}g"] - matchups[f"away_win_pct_{window}g"]
        )
        matchups[f"point_diff_diff_{window}g"] = (
            matchups[f"home_point_diff_avg_{window}g"] - matchups[f"away_point_diff_avg_{window}g"]
        )

    matchups["rest_advantage"] = matchups["home_rest"] - matchups["away_rest"]
    matchups["season_win_pct_diff"] = (
        matchups["home_season_win_pct"] - matchups["away_season_win_pct"]
    )
    matchups["season_point_diff_avg_diff"] = (
        matchups["home_season_point_diff_avg"] - matchups["away_season_point_diff_avg"]
    )

    # ── Totals features (sum-based, not differential) ──────────────────────────
    # Combined offensive output — what do both teams score on average?
    for window in [4, 8, 16]:
        matchups[f"combined_pts_for_{window}g"]     = (
            matchups[f"home_pts_for_avg_{window}g"] + matchups[f"away_pts_for_avg_{window}g"]
        )
        matchups[f"combined_pts_against_{window}g"] = (
            matchups[f"home_pts_against_avg_{window}g"] + matchups[f"away_pts_against_avg_{window}g"]
        )
        # Each team's avg game total (pts_for + pts_against in their games)
        matchups[f"home_game_total_avg_{window}g"] = (
            matchups[f"home_pts_for_avg_{window}g"] + matchups[f"home_pts_against_avg_{window}g"]
        )
        matchups[f"away_game_total_avg_{window}g"] = (
            matchups[f"away_pts_for_avg_{window}g"] + matchups[f"away_pts_against_avg_{window}g"]
        )
    # Season-level combined pace
    matchups["combined_season_pts_for"] = (
        matchups["home_season_point_diff_avg"] + matchups["away_season_point_diff_avg"]
    )
    matchups["combined_rest"] = matchups["home_rest"] + matchups["away_rest"]

    # ── Weather / venue features (totals-focused) ──────────────────────────────
    # Wind and temperature meaningfully suppress passing / kicking efficiency;
    # domes eliminate weather entirely. Use forgiving defaults for missing data
    # (mild conditions ≈ neutral impact) so we never drop games in training.
    roof_series = matchups.get("roof", pd.Series("outdoors", index=matchups.index)).fillna("outdoors")
    matchups["is_dome"] = roof_series.isin(["dome", "closed"]).astype(int)
    # Domes ⇒ no wind, controlled temp. Outdoor NaNs default to mild.
    temp_default = 65.0
    wind_default = 5.0
    temp_col = matchups["temp"] if "temp" in matchups.columns else pd.Series(np.nan, index=matchups.index)
    wind_col = matchups["wind"] if "wind" in matchups.columns else pd.Series(np.nan, index=matchups.index)
    matchups["temp_f"]   = pd.to_numeric(temp_col, errors="coerce")
    matchups["wind_mph"] = pd.to_numeric(wind_col, errors="coerce")
    matchups.loc[matchups["is_dome"] == 1, "temp_f"]   = 72.0
    matchups.loc[matchups["is_dome"] == 1, "wind_mph"] = 0.0
    matchups["temp_f"]   = matchups["temp_f"].fillna(temp_default)
    matchups["wind_mph"] = matchups["wind_mph"].fillna(wind_default)
    # High wind (>15mph) is the main scoring suppressor; cold (<32F) is secondary
    matchups["is_high_wind"]  = (matchups["wind_mph"] > 15).astype(int)
    matchups["is_cold"]       = (matchups["temp_f"] < 32).astype(int)

    # ── QB features ────────────────────────────────────────────────────────────
    if include_qb:
        try:
            from data.nfl_qb_stats import get_qb_features_for_matchups
            qb_df = get_qb_features_for_matchups(seasons)
            qb_cols = [c for c in qb_df.columns if c not in ("season", "week", "team")]

            home_qb = qb_df.rename(columns={"team": "home_team"})
            home_qb = home_qb.rename(columns={c: f"home_{c}" for c in qb_cols})
            matchups = matchups.merge(home_qb, on=["season", "week", "home_team"], how="left")

            away_qb = qb_df.rename(columns={"team": "away_team"})
            away_qb = away_qb.rename(columns={c: f"away_{c}" for c in qb_cols})
            matchups = matchups.merge(away_qb, on=["season", "week", "away_team"], how="left")

            for col in qb_cols:
                matchups[f"qb_{col.replace('qb_','')}_diff"] = (
                    matchups[f"home_{col}"] - matchups[f"away_{col}"]
                )
            # Fill NaN (early season weeks before rolling window is populated)
            qb_feature_cols = (
                [f"home_{c}" for c in qb_cols]
                + [f"away_{c}" for c in qb_cols]
                + [f"qb_{c.replace('qb_','')}_diff" for c in qb_cols]
            )
            matchups[qb_feature_cols] = matchups[qb_feature_cols].fillna(0.0)
            # Totals: combined QB efficiency (higher combined EPA → more scoring)
            for col in qb_cols:
                matchups[f"combined_{col}"] = matchups[f"home_{col}"] + matchups[f"away_{col}"]
            print(f"  QB features merged: {len(qb_cols)} rolling cols per team")
        except Exception as e:
            print(f"  Warning: QB features skipped — {e}")

    # ── Team EPA features (offense + defense, rolling 4/8w) ───────────────────
    try:
        from data.nfl_team_epa import get_team_epa_for_matchups
        epa_df = get_team_epa_for_matchups(seasons)
        epa_cols = [c for c in epa_df.columns if c not in ("season", "week", "team")]

        home_epa = epa_df.rename(columns={"team": "home_team"})
        home_epa = home_epa.rename(columns={c: f"home_{c}" for c in epa_cols})
        matchups = matchups.merge(home_epa, on=["season", "week", "home_team"], how="left")

        away_epa = epa_df.rename(columns={"team": "away_team"})
        away_epa = away_epa.rename(columns={c: f"away_{c}" for c in epa_cols})
        matchups = matchups.merge(away_epa, on=["season", "week", "away_team"], how="left")

        # Diffs use (home - away) — consistent with QB and point-diff features
        for col in epa_cols:
            matchups[f"{col}_diff"] = matchups[f"home_{col}"] - matchups[f"away_{col}"]

        # Totals: combined offensive EPA (both offenses efficient ⇒ higher totals)
        # and combined defensive EPA (both defenses leaky ⇒ higher totals)
        for col in epa_cols:
            matchups[f"combined_{col}"] = matchups[f"home_{col}"] + matchups[f"away_{col}"]

        # NaN handling: early season / new teams — fill with league-neutral 0.0
        # (EPA/play is already zero-centered so 0.0 ≈ league average)
        epa_feature_cols = (
            [f"home_{c}" for c in epa_cols]
            + [f"away_{c}" for c in epa_cols]
            + [f"{c}_diff" for c in epa_cols]
            + [f"combined_{c}" for c in epa_cols]
        )
        matchups[epa_feature_cols] = matchups[epa_feature_cols].fillna(0.0)
        print(f"  Team EPA features merged: {len(epa_cols)} rolling cols per team")
    except Exception as e:
        print(f"  Warning: Team EPA features skipped — {e}")

    # ── Injury features ────────────────────────────────────────────────────────
    if include_injuries:
        try:
            from data.nfl_injuries import get_injury_features_for_matchups
            # Only fetch seasons where injury data is reliable (2015+)
            inj_seasons = [s for s in seasons if s >= 2015]
            if inj_seasons:
                inj_df = get_injury_features_for_matchups(inj_seasons)
                inj_cols = ["injury_score", "qb_injury_impact"]

                home_inj = inj_df.rename(columns={"team": "home_team"})
                home_inj = home_inj.rename(columns={c: f"home_{c}" for c in inj_cols})
                matchups = matchups.merge(home_inj, on=["season", "week", "home_team"], how="left")

                away_inj = inj_df.rename(columns={"team": "away_team"})
                away_inj = away_inj.rename(columns={c: f"away_{c}" for c in inj_cols})
                matchups = matchups.merge(away_inj, on=["season", "week", "away_team"], how="left")

                for col in inj_cols:
                    matchups[f"{col}_diff"] = (
                        matchups[f"away_{col}"] - matchups[f"home_{col}"]
                    )
                inj_feature_cols = (
                    [f"home_{c}" for c in inj_cols]
                    + [f"away_{c}" for c in inj_cols]
                    + [f"{c}_diff" for c in inj_cols]
                )
                matchups[inj_feature_cols] = matchups[inj_feature_cols].fillna(0.0)
                # Totals: combined injury load suppresses scoring
                for col in inj_cols:
                    matchups[f"combined_{col}"] = matchups[f"home_{col}"] + matchups[f"away_{col}"]
                print(f"  Injury features merged")
        except Exception as e:
            print(f"  Warning: Injury features skipped — {e}")

    cache_path = NFL_RAW / "matchups.csv"
    matchups.to_csv(cache_path, index=False)
    return matchups


def get_current_team_stats(seasons: list = None) -> pd.DataFrame:
    """Most recent rolling stats for every team — used for live predictions."""
    if seasons is None:
        current_year = pd.Timestamp.now().year
        # NFL season spans two calendar years; use recent seasons
        seasons = list(range(current_year - 4, current_year + 1))
    log = build_team_game_log(seasons)
    latest = log.sort_values("date").groupby("team").last().reset_index()
    return latest


if __name__ == "__main__":
    df = build_matchup_features(list(range(2015, datetime.now().year + 1)))
    print(f"Built {len(df)} matchup rows")
    print(df.dtypes)
    print(df.head())
