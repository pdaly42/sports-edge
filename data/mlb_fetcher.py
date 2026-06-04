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
        df = pd.read_csv(cache_path, parse_dates=["date"])
        # If cached before pitcher columns were added, re-fetch
        if "win_pitcher" not in df.columns:
            cache_path.unlink()
        else:
            return df

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
            # Pitcher who got the decision — last name only from BRef
            "win_pitcher":   str(r.get("Win", "")).strip(),
            "loss_pitcher":  str(r.get("Loss", "")).strip(),
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
    Includes starting pitcher stats (ERA, WHIP, K/9, K/BB) sourced from BRef season data.
    Home team perspective — target: home_win (1 = home team won).
    """
    from data.mlb_pitcher_stats import build_season_pitcher_lookup, lookup_pitcher, pitcher_stats_or_median

    log = build_team_game_log(seasons)
    if log.empty:
        print("No data fetched.")
        return pd.DataFrame()

    # Build pitcher lookup once for all seasons
    print("  Loading pitcher season stats...")
    pitcher_lookup = build_season_pitcher_lookup(seasons)

    home_log = log[log["is_home"] == 1].copy()
    away_log = log[log["is_home"] == 0].copy()

    stat_cols = [c for c in log.columns if any(
        c.startswith(p) for p in ["win_pct", "runs_for_avg", "runs_against_avg", "run_diff_avg"]
    )] + ["days_rest"]

    pitcher_base_cols = ["win_pitcher", "loss_pitcher"]

    home_feats = home_log[["date", "team", "opponent", "win", "season"]
                           + stat_cols + pitcher_base_cols].copy()
    home_feats.columns = (["date", "home_team", "away_team", "home_win", "season"]
                          + [f"home_{c}" for c in stat_cols]
                          + ["home_win_pitcher", "home_loss_pitcher"])

    away_feats = away_log[["date", "team"] + stat_cols + pitcher_base_cols].copy()
    away_feats.columns = (["date", "away_team"]
                          + [f"away_{c}" for c in stat_cols]
                          + ["away_win_pitcher", "away_loss_pitcher"])

    matchups = home_feats.merge(away_feats, on=["date", "away_team"])

    # Differential team stats
    for window in [5, 10, 20]:
        matchups[f"win_pct_diff_{window}g"]  = (
            matchups[f"home_win_pct_{window}g"]      - matchups[f"away_win_pct_{window}g"])
        matchups[f"run_diff_diff_{window}g"] = (
            matchups[f"home_run_diff_avg_{window}g"] - matchups[f"away_run_diff_avg_{window}g"])

    matchups["rest_advantage"] = matchups["home_days_rest"] - matchups["away_days_rest"]

    # ── Add pitcher features ──────────────────────────────────────
    # The Win pitcher is the home team's starter when home won (and vice versa)
    # The Loss pitcher is the home team's starter when home lost
    pitcher_cols = ["era", "whip", "k_per_9", "k_bb_ratio", "innings_pitched"]

    for side in ("home", "away"):
        for col in pitcher_cols:
            matchups[f"{side}_starter_{col}"] = np.nan

    matched = 0
    for idx, row in matchups.iterrows():
        season = int(row["season"])
        sdict  = pitcher_lookup.get(season, {})

        # Identify which pitcher name corresponds to home vs away starter
        if row["home_win"] == 1:
            home_p_name = row["home_win_pitcher"]
            away_p_name = row["home_loss_pitcher"]
        else:
            home_p_name = row["home_loss_pitcher"]
            away_p_name = row["home_win_pitcher"]

        for side, p_name, team in [
            ("home", home_p_name, row["home_team"]),
            ("away", away_p_name, row["away_team"]),
        ]:
            raw_stats = lookup_pitcher(sdict, p_name, team)
            stats = pitcher_stats_or_median(raw_stats)
            for col in pitcher_cols:
                matchups.at[idx, f"{side}_starter_{col}"] = stats[col]
            if raw_stats:
                matched += 1

    total = len(matchups) * 2
    print(f"  Pitcher stats matched: {matched}/{total} ({100*matched//total}%)")

    # Differential pitcher features
    matchups["starter_era_diff"]    = matchups["home_starter_era"]    - matchups["away_starter_era"]
    matchups["starter_whip_diff"]   = matchups["home_starter_whip"]   - matchups["away_starter_whip"]
    matchups["starter_k9_diff"]     = matchups["home_starter_k_per_9"]- matchups["away_starter_k_per_9"]

    # Drop the raw pitcher name columns — not useful as model features
    matchups.drop(columns=["home_win_pitcher","home_loss_pitcher",
                            "away_win_pitcher","away_loss_pitcher"], inplace=True)

    cache_path = MLB_RAW / "matchups_with_pitchers.csv"
    matchups.to_csv(cache_path, index=False)
    print(f"Built {len(matchups)} MLB matchup rows with pitcher features")
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
