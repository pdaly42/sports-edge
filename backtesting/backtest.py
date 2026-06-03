"""
Backtesting engine: simulate betting on historical games using model probabilities.
Supports flat staking and fractional Kelly criterion.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

from config.settings import STARTING_BANKROLL, KELLY_FRACTION, MIN_EDGE_THRESHOLD
from utils.odds import remove_vig, expected_value, kelly_fraction as kelly_calc, american_to_implied


@dataclass
class BetRecord:
    date: str
    home_team: str
    away_team: str
    bet_side: str          # "home" or "away"
    model_prob: float
    market_odds: float     # American odds for the side we're betting
    no_vig_prob: float     # market's fair probability for our side
    edge: float            # model_prob - no_vig_prob
    ev: float
    kelly: float
    stake: float
    outcome: int           # 1 = win, 0 = loss
    pnl: float


def run_backtest(
    matchups: pd.DataFrame,
    predictions: np.ndarray,          # model home-win probabilities, aligned to matchups rows
    odds_df: pd.DataFrame | None = None,  # optional: historical odds (date, home_team, away_team, home_odds, away_odds)
    bet_type: str = "moneyline",
    staking: str = "kelly",           # "flat" or "kelly"
    flat_unit: float = 20.0,
    min_edge: float = MIN_EDGE_THRESHOLD,
) -> pd.DataFrame:
    """
    Simulate bets. If odds_df is None, uses synthetic -110/-110 lines (standard spread).
    Returns a DataFrame of all bet records plus a summary.
    """
    df = matchups.copy().reset_index(drop=True)
    df["model_home_prob"] = predictions

    if odds_df is not None:
        df = df.merge(odds_df, on=["date", "home_team", "away_team"], how="left")
    else:
        # Default to -110 both sides (4.55% vig) when no odds data available
        df["home_odds"] = -110
        df["away_odds"] = -110

    records = []
    bankroll = STARTING_BANKROLL

    for _, row in df.iterrows():
        if pd.isna(row.get("home_odds")) or pd.isna(row.get("away_odds")):
            continue

        home_fair, away_fair = remove_vig(row["home_odds"], row["away_odds"])
        model_home = row["model_home_prob"]
        model_away = 1 - model_home

        candidates = [
            ("home", model_home, home_fair, row["home_odds"]),
            ("away", model_away, away_fair, row["away_odds"]),
        ]

        for side, model_p, fair_p, odds in candidates:
            edge = model_p - fair_p
            if edge < min_edge:
                continue

            ev = expected_value(model_p, odds)
            k = kelly_calc(model_p, odds) * KELLY_FRACTION

            if staking == "kelly":
                stake = bankroll * k
            else:
                stake = flat_unit

            stake = min(stake, bankroll * 0.05)  # hard cap: never risk more than 5% on one bet
            if stake <= 0:
                continue

            if odds > 0:
                win_payout = stake * (odds / 100)
            else:
                win_payout = stake * (100 / abs(odds))

            actual_home_win = int(row["home_win"])
            won = (side == "home" and actual_home_win == 1) or (side == "away" and actual_home_win == 0)
            pnl = win_payout if won else -stake
            bankroll += pnl

            records.append(BetRecord(
                date=str(row["date"]),
                home_team=row["home_team"],
                away_team=row["away_team"],
                bet_side=side,
                model_prob=round(model_p, 4),
                market_odds=odds,
                no_vig_prob=round(fair_p, 4),
                edge=round(edge, 4),
                ev=round(ev, 4),
                kelly=round(k, 4),
                stake=round(stake, 2),
                outcome=int(won),
                pnl=round(pnl, 2),
            ))

    bets_df = pd.DataFrame([vars(r) for r in records])

    if bets_df.empty:
        print("No bets met the edge threshold.")
        return bets_df

    bets_df["cumulative_pnl"] = bets_df["pnl"].cumsum()
    bets_df["bankroll"] = STARTING_BANKROLL + bets_df["cumulative_pnl"]

    _print_summary(bets_df)
    return bets_df


def _print_summary(df: pd.DataFrame) -> None:
    total_bets = len(df)
    wins = df["outcome"].sum()
    total_staked = df["stake"].sum()
    total_pnl = df["pnl"].sum()
    roi = total_pnl / total_staked if total_staked > 0 else 0

    print("\n=== Backtest Summary ===")
    print(f"Total bets:    {total_bets}")
    print(f"Win rate:      {wins/total_bets:.1%}")
    print(f"Total staked:  ${total_staked:,.2f}")
    print(f"Total P&L:     ${total_pnl:,.2f}")
    print(f"ROI:           {roi:.2%}")
    print(f"Final bankroll: ${STARTING_BANKROLL + total_pnl:,.2f}")


def plot_bankroll(df: pd.DataFrame, title: str = "Bankroll Over Time") -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["bankroll"], linewidth=1.5)
    plt.axhline(y=STARTING_BANKROLL, color="gray", linestyle="--", alpha=0.5)
    plt.title(title)
    plt.xlabel("Bet #")
    plt.ylabel("Bankroll ($)")
    plt.tight_layout()
    plt.show()


def edge_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4))
    plt.hist(df["edge"], bins=30, edgecolor="black")
    plt.axvline(x=0, color="red", linestyle="--")
    plt.title("Distribution of Model Edge on Placed Bets")
    plt.xlabel("Edge (model prob - no-vig market prob)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
