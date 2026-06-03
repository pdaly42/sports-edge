"""
Main pipeline: fetch data → engineer features → train model → backtest.
Run: python run_pipeline.py --sport nba --seasons 2019 2020 2021 2022 2023 2024
"""

import argparse
import sys
sys.path.insert(0, ".")

import pandas as pd
from models.trainer import train, predict_proba
from backtesting.backtest import run_backtest, plot_bankroll, edge_distribution


def run_nba(seasons: list[int]) -> None:
    from data.nba_fetcher import build_matchup_features

    print(f"Building NBA matchup features for seasons: {seasons}")
    matchups = build_matchup_features(seasons)
    matchups = matchups.dropna(subset=["home_win"]).reset_index(drop=True)

    # Train on all but the last season; test on the last
    cutoff_season = max(seasons)
    train_df = matchups[matchups["season"] < cutoff_season].copy()
    test_df = matchups[matchups["season"] == cutoff_season].copy()

    print(f"\nTraining on {len(train_df)} games, testing on {len(test_df)} games")

    bundle = train(train_df, sport="nba", model_type="xgb")

    test_df = test_df.dropna(subset=bundle["features"])
    preds = predict_proba(bundle, test_df)

    print("\nRunning backtest on held-out season...")
    bets = run_backtest(test_df, preds, staking="kelly")

    if not bets.empty:
        plot_bankroll(bets, title=f"NBA {cutoff_season} Season — Kelly Staking Backtest")
        edge_distribution(bets)

        top_edges = bets.nlargest(10, "edge")[
            ["date", "home_team", "away_team", "bet_side", "model_prob", "no_vig_prob", "edge", "ev", "pnl"]
        ]
        print("\nTop 10 highest-edge bets:")
        print(top_edges.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Sports edge pipeline")
    parser.add_argument("--sport", default="nba", choices=["nba", "nfl", "mlb"])
    parser.add_argument("--seasons", nargs="+", type=int, default=list(range(2019, 2025)))
    args = parser.parse_args()

    if args.sport == "nba":
        run_nba(args.seasons)
    else:
        print(f"{args.sport.upper()} fetcher not yet implemented — NFL and MLB coming next.")


if __name__ == "__main__":
    main()
