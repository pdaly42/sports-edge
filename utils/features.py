"""Shared feature engineering utilities used across sports."""

import pandas as pd
import numpy as np


def rolling_avg(df: pd.DataFrame, col: str, window: int, group_col: str = "team") -> pd.Series:
    """Rolling mean of col per team, shifted to avoid leakage."""
    return (
        df.groupby(group_col)[col]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
    )


def days_rest(df: pd.DataFrame, date_col: str = "date", team_col: str = "team") -> pd.Series:
    """Days since each team's last game."""
    df = df.sort_values([team_col, date_col])
    return df.groupby(team_col)[date_col].transform(lambda x: x.diff().dt.days)


def home_away_split(df: pd.DataFrame, stat_col: str, home_col: str = "is_home") -> pd.DataFrame:
    """Add home/away rolling averages for a stat column."""
    out = df.copy()
    out[f"{stat_col}_home_avg"] = (
        df[df[home_col] == 1]
        .groupby("team")[stat_col]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    )
    out[f"{stat_col}_away_avg"] = (
        df[df[home_col] == 0]
        .groupby("team")[stat_col]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    )
    return out


def win_pct(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Rolling win percentage per team."""
    return rolling_avg(df, "win", window)


def encode_matchup(home_team: str, away_team: str) -> str:
    return f"{away_team}@{home_team}"
