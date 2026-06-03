"""
Fetch NBA game logs via basketball-reference-web-scraper.
Saves raw data to data/raw/nba/ and returns engineered feature DataFrames.
"""

import pandas as pd
import numpy as np
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import Team

from config.settings import RAW_DIR
from utils.features import days_rest, rolling_avg

NBA_RAW = RAW_DIR / "nba"
NBA_RAW.mkdir(parents=True, exist_ok=True)


def fetch_season_games(season: int) -> pd.DataFrame:
    """
    Fetch all games for a given NBA season end year (e.g. 2024 = 2023-24).
    Results cached to disk.
    """
    cache_path = NBA_RAW / f"games_{season}.csv"
    if cache_path.exists():
        print(f"  Loading cached NBA {season}")
        return pd.read_csv(cache_path, parse_dates=["date"])

    print(f"  Fetching NBA {season} from basketball-reference...")
    time.sleep(1.5)  # be polite to the server

    try:
        games = client.season_schedule(season_end_year=season)
    except Exception as e:
        print(f"  Error fetching {season}: {e}")
        return pd.DataFrame()

    rows = []
    for g in games:
        home_score = g.get("home_team_score")
        away_score = g.get("away_team_score")
        if home_score is None or away_score is None:
            continue
        rows.append({
            "date": pd.to_datetime(g["start_time"]).tz_localize(None).normalize(),
            "home_team": g["home_team"].value if isinstance(g["home_team"], Team) else str(g["home_team"]),
            "away_team": g["away_team"].value if isinstance(g["away_team"], Team) else str(g["away_team"]),
            "home_score": int(home_score),
            "away_score": int(away_score),
        })

    df = pd.DataFrame(rows).drop_duplicates()
    df.to_csv(cache_path, index=False)
    print(f"  Saved {len(df)} games for {season}")
    return df


def build_team_game_log(seasons: list) -> pd.DataFrame:
    """
    Per-team-per-game log with engineered features.
    Each game appears twice: home perspective and away perspective.
    """
    frames = []
    for s in seasons:
        df = fetch_season_games(s)
        if df.empty:
            continue
        df["season"] = s

        home = df.copy()
        home["team"] = home["home_team"]
        home["opponent"] = home["away_team"]
        home["points_for"] = home["home_score"]
        home["points_against"] = home["away_score"]
        home["is_home"] = 1

        away = df.copy()
        away["team"] = away["away_team"]
        away["opponent"] = away["home_team"]
        away["points_for"] = away["away_score"]
        away["points_against"] = away["home_score"]
        away["is_home"] = 0

        combined = pd.concat([home, away], ignore_index=True)
        combined["win"] = (combined["points_for"] > combined["points_against"]).astype(int)
        combined["point_diff"] = combined["points_for"] - combined["points_against"]
        frames.append(combined[["date","season","team","opponent","is_home",
                                 "points_for","points_against","win","point_diff"]])

    log = pd.concat(frames).sort_values(["team", "date"]).reset_index(drop=True)

    for window in [5, 10, 20]:
        log[f"win_pct_{window}g"] = rolling_avg(log, "win", window)
        log[f"pts_for_avg_{window}g"] = rolling_avg(log, "points_for", window)
        log[f"pts_against_avg_{window}g"] = rolling_avg(log, "points_against", window)
        log[f"point_diff_avg_{window}g"] = rolling_avg(log, "point_diff", window)

    log["days_rest"] = days_rest(log)
    return log


def build_matchup_features(seasons: list) -> pd.DataFrame:
    """
    Per-game matchup DataFrame suitable for model training.
    Target column: home_win (1 = home won).
    """
    log = build_team_game_log(seasons)

    home_log = log[log["is_home"] == 1].copy()
    away_log = log[log["is_home"] == 0].copy()

    stat_cols = [c for c in log.columns if any(
        c.startswith(p) for p in ["win_pct", "pts_for_avg", "pts_against_avg", "point_diff_avg"]
    )] + ["days_rest"]

    home_feats = home_log[["date", "team", "opponent", "win", "season"] + stat_cols].copy()
    home_feats.columns = (["date", "home_team", "away_team", "home_win", "season"]
                          + [f"home_{c}" for c in stat_cols])

    away_feats = away_log[["date", "team"] + stat_cols].copy()
    away_feats.columns = ["date", "away_team"] + [f"away_{c}" for c in stat_cols]

    matchups = home_feats.merge(away_feats, on=["date", "away_team"])

    for window in [5, 10, 20]:
        matchups[f"win_pct_diff_{window}g"] = (
            matchups[f"home_win_pct_{window}g"] - matchups[f"away_win_pct_{window}g"])
        matchups[f"point_diff_diff_{window}g"] = (
            matchups[f"home_point_diff_avg_{window}g"] - matchups[f"away_point_diff_avg_{window}g"])

    matchups["rest_advantage"] = matchups["home_days_rest"] - matchups["away_days_rest"]

    cache_path = NBA_RAW / "matchups.csv"
    matchups.to_csv(cache_path, index=False)
    return matchups


def get_current_team_stats(seasons: list = None) -> pd.DataFrame:
    """
    Return the most recent rolling stats for every team — used for today's predictions.
    """
    if seasons is None:
        seasons = list(range(2022, 2026))
    log = build_team_game_log(seasons)
    # Most recent row per team
    latest = log.sort_values("date").groupby("team").last().reset_index()
    return latest


if __name__ == "__main__":
    df = build_matchup_features(list(range(2019, 2025)))
    print(f"Built {len(df)} matchup rows")
    print(df.head())
