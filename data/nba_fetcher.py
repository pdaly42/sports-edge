"""
Fetch NBA game logs and box scores via basketball-reference-web-scraper.
Saves raw data to data/raw/nba/ and returns a cleaned DataFrame.
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import OutputType

from config.settings import RAW_DIR
from utils.features import days_rest, rolling_avg, win_pct

NBA_RAW = RAW_DIR / "nba"
NBA_RAW.mkdir(parents=True, exist_ok=True)


def fetch_season_games(season: int) -> pd.DataFrame:
    """
    Fetch all games for a given NBA season end year (e.g. 2024 = 2023-24).
    Results cached to disk.
    """
    cache_path = NBA_RAW / f"games_{season}.csv"
    if cache_path.exists():
        print(f"Loading cached NBA {season} games")
        return pd.read_csv(cache_path, parse_dates=["date"])

    print(f"Fetching NBA {season} season from basketball-reference...")
    rows = []
    for month in range(1, 13):
        try:
            games = client.season_schedule(season_end_year=season)
            for g in games:
                rows.append({
                    "date": g["start_time"].date(),
                    "home_team": g["home_team"].value,
                    "away_team": g["away_team"].value,
                    "home_score": g.get("home_team_score"),
                    "away_score": g.get("away_team_score"),
                })
            break
        except Exception as e:
            print(f"  month {month} error: {e}")
            time.sleep(2)

    df = pd.DataFrame(rows).drop_duplicates()
    df["date"] = pd.to_datetime(df["date"])
    df.to_csv(cache_path, index=False)
    return df


def build_team_game_log(seasons: list[int]) -> pd.DataFrame:
    """
    Build a per-team-per-game log with engineered features from raw game data.
    Each game appears twice: once for each team (home and away perspective).
    """
    frames = []
    for s in seasons:
        df = fetch_season_games(s)
        df = df.dropna(subset=["home_score", "away_score"])
        df["season"] = s

        # Flatten to per-team rows
        home = df.assign(
            team=df["home_team"],
            opponent=df["away_team"],
            points_for=df["home_score"],
            points_against=df["away_score"],
            is_home=1,
        )
        away = df.assign(
            team=df["away_team"],
            opponent=df["home_team"],
            points_for=df["away_score"],
            points_against=df["home_score"],
            is_home=0,
        )
        combined = pd.concat([home, away], ignore_index=True)
        combined["win"] = (combined["points_for"] > combined["points_against"]).astype(int)
        combined["point_diff"] = combined["points_for"] - combined["points_against"]
        frames.append(combined)

    log = pd.concat(frames).sort_values(["team", "date"]).reset_index(drop=True)

    # Feature engineering
    for window in [5, 10, 20]:
        log[f"win_pct_{window}g"] = rolling_avg(log, "win", window)
        log[f"pts_for_avg_{window}g"] = rolling_avg(log, "points_for", window)
        log[f"pts_against_avg_{window}g"] = rolling_avg(log, "points_against", window)
        log[f"point_diff_avg_{window}g"] = rolling_avg(log, "point_diff", window)

    log["days_rest"] = days_rest(log)

    return log


def build_matchup_features(seasons: list[int]) -> pd.DataFrame:
    """
    Build a per-game matchup DataFrame suitable for model training.
    Target: home_win (1 = home team won).
    """
    log = build_team_game_log(seasons)

    home_log = log[log["is_home"] == 1].copy()
    away_log = log[log["is_home"] == 0].copy()

    feature_cols = [c for c in log.columns if any(
        c.startswith(p) for p in ["win_pct", "pts_for", "pts_against", "point_diff"]
    )] + ["days_rest"]

    home_feats = home_log[["date", "team", "opponent", "win", "season"] + feature_cols].copy()
    home_feats.columns = (
        ["date", "home_team", "away_team", "home_win", "season"]
        + [f"home_{c}" for c in feature_cols]
    )

    away_feats = away_log[["date", "team"] + feature_cols].copy()
    away_feats.columns = ["date", "away_team"] + [f"away_{c}" for c in feature_cols]

    matchups = home_feats.merge(away_feats, on=["date", "away_team"])

    # Differential features — often more predictive than raw values
    for window in [5, 10, 20]:
        matchups[f"win_pct_diff_{window}g"] = (
            matchups[f"home_win_pct_{window}g"] - matchups[f"away_win_pct_{window}g"]
        )
        matchups[f"point_diff_diff_{window}g"] = (
            matchups[f"home_point_diff_avg_{window}g"] - matchups[f"away_point_diff_avg_{window}g"]
        )

    matchups["rest_advantage"] = matchups["home_days_rest"] - matchups["away_days_rest"]

    cache_path = RAW_DIR / "nba" / "matchups.csv"
    matchups.to_csv(cache_path, index=False)
    print(f"Saved {len(matchups)} matchup rows to {cache_path}")
    return matchups


if __name__ == "__main__":
    df = build_matchup_features(list(range(2018, 2025)))
    print(df.shape)
    print(df.head())
