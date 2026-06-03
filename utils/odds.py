"""Utilities for converting between odds formats and implied probabilities."""

import numpy as np


def american_to_implied(american_odds: float) -> float:
    """Convert American odds to implied probability (no vig removed)."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def implied_to_american(prob: float) -> float:
    """Convert true probability to American odds."""
    if prob >= 0.5:
        return -(prob / (1 - prob)) * 100
    else:
        return ((1 - prob) / prob) * 100


def remove_vig(home_odds: float, away_odds: float) -> tuple[float, float]:
    """
    Remove the vig from a two-sided market.
    Returns (home_true_prob, away_true_prob) that sum to 1.0.
    """
    home_implied = american_to_implied(home_odds)
    away_implied = american_to_implied(away_odds)
    total = home_implied + away_implied
    return home_implied / total, away_implied / total


def expected_value(model_prob: float, american_odds: float) -> float:
    """
    Compute expected value of a bet.
    Returns EV as a fraction of the bet amount (e.g. 0.05 = 5% EV).
    """
    if american_odds > 0:
        payout = american_odds / 100
    else:
        payout = 100 / abs(american_odds)

    return model_prob * payout - (1 - model_prob)


def kelly_fraction(model_prob: float, american_odds: float) -> float:
    """
    Full Kelly criterion: optimal fraction of bankroll to bet.
    Returns 0 if bet has negative EV.
    """
    if american_odds > 0:
        b = american_odds / 100
    else:
        b = 100 / abs(american_odds)

    q = 1 - model_prob
    kelly = (b * model_prob - q) / b
    return max(kelly, 0.0)


def edge(model_prob: float, market_odds: float) -> float:
    """
    Edge = model's implied probability minus market's no-vig probability.
    Positive edge means you have an advantage.
    Requires the fair-side odds (already vig-removed or single side passed directly).
    """
    market_implied = american_to_implied(market_odds)
    return model_prob - market_implied
