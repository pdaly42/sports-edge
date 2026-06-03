"""
Sharp money detector: identify games where significant line movement
suggests professional ("sharp") bettors are on a side.

Sharp action is one of the most reliable edges in sports betting.
When sharps move a line significantly, fading the public and following
the sharp side historically outperforms.
"""

import pandas as pd
import numpy as np


def detect_reverse_line_movement(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reverse Line Movement (RLM): public % is on one side but the line
    moves against them — sharps are likely on the other side.

    Expected columns: date, home_team, away_team,
                      open_home_odds, close_home_odds,
                      public_home_pct (% of public bets on home)
    """
    df = odds_df.copy()

    # Convert to implied probabilities
    def imp(o):
        return 100 / (o + 100) if o > 0 else abs(o) / (abs(o) + 100)

    df["open_home_implied"] = df["open_home_odds"].apply(imp)
    df["close_home_implied"] = df["close_home_odds"].apply(imp)

    # Line movement direction (positive = home got more expensive)
    df["line_move"] = df["close_home_implied"] - df["open_home_implied"]

    # RLM: public likes home (>55%) but line moved away from home
    df["rlm_away"] = (df["public_home_pct"] > 55) & (df["line_move"] < -0.02)
    # RLM: public likes away (<45% home) but line moved toward home
    df["rlm_home"] = (df["public_home_pct"] < 45) & (df["line_move"] > 0.02)

    df["sharp_side"] = None
    df.loc[df["rlm_home"], "sharp_side"] = "home"
    df.loc[df["rlm_away"], "sharp_side"] = "away"

    return df


def steam_move_filter(odds_df: pd.DataFrame, threshold_minutes: int = 10) -> pd.DataFrame:
    """
    Steam move: rapid line movement across multiple books in a short window.
    Requires a timed odds feed with multiple books per game.

    Expected columns: game_id, book, timestamp, home_odds
    Returns games flagged as steam moves.
    """
    df = odds_df.sort_values(["game_id", "timestamp"])
    results = []

    for game_id, group in df.groupby("game_id"):
        for book in group["book"].unique():
            book_data = group[group["book"] == book].sort_values("timestamp")
            if len(book_data) < 2:
                continue
            move = book_data["home_odds"].iloc[-1] - book_data["home_odds"].iloc[0]
            elapsed = (book_data["timestamp"].iloc[-1] - book_data["timestamp"].iloc[0]).total_seconds() / 60
            if abs(move) >= 10 and elapsed <= threshold_minutes:
                results.append({"game_id": game_id, "book": book, "move": move, "minutes": elapsed})

    return pd.DataFrame(results)
