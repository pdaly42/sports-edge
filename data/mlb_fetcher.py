"""
Fetch MLB game logs via pybaseball (Baseball Reference).
One request per team per season — results cached to data/raw/mlb/.
Builds the same style of matchup DataFrame as nba_fetcher.py.
"""

import sys, os, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from pybaseball import schedule_and_record

from config.settings import RAW_DIR
from utils.features import rolling_avg, days_rest

MLB_RAW = RAW_DIR / "mlb"
MLB_RAW.mkdir(parents=True, exist_ok=True)

# All 30 MLB team abbreviations (Baseball Reference format)
ALL_TEAMS = [
    "ATL","MIA","NYM","PHI","WSN",
    "CHC","CIN","MIL","PIT","STL",
    "ARI","COL","LAD","SDP","SFG",
    "BAL","BOS","NYY","TBR","TOR",
    "CHW","CLE","DET","KCR","MIN",
    "HOU","LAA","OAK","SEA","TEX",
]

# Odds API team name → Baseball Reference abbreviation
TEAM_MAP = {
    "Atlanta Braves":        "ATL",
    "Miami Marlins":         "MIA",
    "New York Mets":         "NYM",
    "Philadelphia Phillies": "PHI",
    "Washington Nationals":  "WSN",
    "Chicago Cubs":          "CHC",
    "Cincinnati Reds":       "CIN",
    "Milwaukee Brewers":     "MIL",
    "Pittsburgh Pirates":    "PIT",
    "St. Louis Cardinals":   "STL",
    "Arizona Diamondbacks":  "ARI",
    "Colorado Rockies":      "COL",
    "Los Angeles Dodgers":   "LAD",
    "San Diego Padres":      "SDP",
    "San Francisco Giants":  "SFG",
    "Baltimore Orioles":     "BAL",
    "Boston Red Sox":        "BOS",
    "New York Yankees":      "NYY",
    "Tampa Bay Rays":        "TBR",
    "Toronto Blue Jays":     "TOR",
    "Chicago White Sox":     "CHW",
    "Cleveland Guardians":   "CLE",
    "Detroit Tigers":        "DET",
    "Kansas City Royals":    "KCR",
    "Minnesota Twins":       "MIN",
    "Houston Astros":        "HOU",
    "Los Angeles Angels":    "LAA",
    "Oakland Athletics":     "OAK",
    "Seattle Mariners":      "SEA",
    "Texas Rangers":         "TEX",
}


def fetch_team_season(team: str, season: int) -> pd.DataFrame:
    """
    Fetch one team's game log for one season. Cached to disk after first pull.
    Returns a clean DataFrame with: date, team, opponent, runs_for, runs_against,
    is_home, win, run_diff, season.
    """
    cache_path = MLB_RAW / f"{team}_{season}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["date"])

    print(f"    Fetching {team} {season}...")
    time.sleep(2.5)  # respectful throttle for Baseball Reference
    try:
        raw = schedule_and_record(season, team)
    except Exception as e:
        print(f"    Error {team} {season}: {e}")
        return pd.DataFrame()

    # Drop rows without a result (postponed, future games)
    raw = raw.dropna(subset=["W/L", "R", "RA"])
    raw = raw[raw["W/L"].isin(["W", "L", "W-wo", "L-wo"])]

    rows = []
    for _, r in raw.iterrows():
        # Parse date — format is "Thursday, Mar 30", need to add year
        try:
            date = pd.to_datetime(f"{r['Date'].split(', ',1)[1]} {season}", format="%b %d %Y")
        except Exception:
            continue

        is_home = 1 if str(r.get("Home_Away", "")).strip() != "@" else 0
        win     = 1 if str(r["W/L"]).startswith("W") else 0

        rows.append({
            "date":          date,
            "season":        season,
            "team":          team,
            "opponent":      str(r["Opp"]).strip(),
            "runs_for":      float(r["R"]),
            "runs_against":  float(r["RA"]),
            "is_home":       is_home,
            "win":           win,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["run_diff"] = df["runs_for"] - df["runs_against"]
        df.to_csv(cache_path, index=False)

    return df


def build_team_game_log(seasons: list) -> pd.DataFrame:
    """
    Build a per-team-per-game log for all 30 teams across given seasons.
    Adds rolling stats (5/10/20 game windows) and rest days.
    """
    frames = []
    for season in seasons:
        print(f"  Season {season}:")
        for team in ALL_TEAMS:
            df = fetch_team_season(team, season)
            if not df.empty:
                frames.append(df)

    if not frames:
        return pd.DataFrame()

    log = pd.concat(frames, ignore_index=True)
    log = log.sort_values(["team", "date"]).reset_index(drop=True)

    # Rolling stats — shifted to avoid leakage (only past games)
    for window in [5, 10, 20]:
        log[f"win_pct_{window}g"]          = rolling_avg(log, "win",          window)
        log[f"runs_for_avg_{window}g"]     = rolling_avg(log, "runs_for",     window)
        log[f"runs_against_avg_{window}g"] = rolling_avg(log, "runs_against", window)
        log[f"run_diff_avg_{window}g"]     = rolling_avg(log, "run_diff",     window)

    log["days_rest"] = days_rest(log)
    return log


def build_matchup_features(seasons: list) -> pd.DataFrame:
    """
    Flatten the team log into one row per game with both teams' stats side by side.
    Home team perspective — target: home_win (1 = home team won).
    """
    log = build_team_game_log(seasons)
    if log.empty:
        print("No data fetched.")
        return pd.DataFrame()

    home_log = log[log["is_home"] == 1].copy()
    away_log = log[log["is_home"] == 0].copy()

    stat_cols = [c for c in log.columns if any(
        c.startswith(p) for p in ["win_pct", "runs_for_avg", "runs_against_avg", "run_diff_avg"]
    )] + ["days_rest"]

    home_feats = home_log[["date", "team", "opponent", "win", "season"] + stat_cols].copy()
    home_feats.columns = (["date", "home_team", "away_team", "home_win", "season"]
                          + [f"home_{c}" for c in stat_cols])

    away_feats = away_log[["date", "team"] + stat_cols].copy()
    away_feats.columns = ["date", "away_team"] + [f"away_{c}" for c in stat_cols]

    matchups = home_feats.merge(away_feats, on=["date", "away_team"])

    # Differential features — how much better is home vs away right now
    for window in [5, 10, 20]:
        matchups[f"win_pct_diff_{window}g"]  = (
            matchups[f"home_win_pct_{window}g"]      - matchups[f"away_win_pct_{window}g"])
        matchups[f"run_diff_diff_{window}g"] = (
            matchups[f"home_run_diff_avg_{window}g"] - matchups[f"away_run_diff_avg_{window}g"])

    matchups["rest_advantage"] = matchups["home_days_rest"] - matchups["away_days_rest"]

    cache_path = MLB_RAW / "matchups.csv"
    matchups.to_csv(cache_path, index=False)
    print(f"Built {len(matchups)} MLB matchup rows across {len(seasons)} seasons")
    return matchups


def get_current_team_stats(seasons: list = None) -> pd.DataFrame:
    """Most recent rolling stats per team — used for today's predictions."""
    if seasons is None:
        seasons = list(range(2023, 2026))
    log = build_team_game_log(seasons)
    return log.sort_values("date").groupby("team").last().reset_index()


if __name__ == "__main__":
    df = build_matchup_features(list(range(2019, 2025)))
    print(df.shape)
    print(df.head())
