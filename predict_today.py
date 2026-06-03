"""
Daily prediction script.
1. Fetches today's NBA games + odds from the-odds-api.com
2. Pulls current team stats from basketball-reference
3. Builds matchup features and runs the trained model
4. Outputs predictions.json — loaded by dashboard.html
Run: python3 predict_today.py --api-key YOUR_KEY
     python3 predict_today.py  (reads ODDS_API_KEY from .env)
"""

import sys, os, json, argparse, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import numpy as np
from datetime import date, datetime, timezone
from pathlib import Path

from config.settings import RAW_DIR
from models.trainer import load_model, predict_proba
from utils.odds import remove_vig, expected_value, kelly_fraction as kelly_calc
from utils.features import rolling_avg, days_rest

# ── Team name mapping: odds API → basketball-reference format ──
TEAM_MAP = {
    "Atlanta Hawks": "ATLANTA HAWKS",
    "Boston Celtics": "BOSTON CELTICS",
    "Brooklyn Nets": "BROOKLYN NETS",
    "Charlotte Hornets": "CHARLOTTE HORNETS",
    "Chicago Bulls": "CHICAGO BULLS",
    "Cleveland Cavaliers": "CLEVELAND CAVALIERS",
    "Dallas Mavericks": "DALLAS MAVERICKS",
    "Denver Nuggets": "DENVER NUGGETS",
    "Detroit Pistons": "DETROIT PISTONS",
    "Golden State Warriors": "GOLDEN STATE WARRIORS",
    "Houston Rockets": "HOUSTON ROCKETS",
    "Indiana Pacers": "INDIANA PACERS",
    "Los Angeles Clippers": "LOS ANGELES CLIPPERS",
    "Los Angeles Lakers": "LOS ANGELES LAKERS",
    "Memphis Grizzlies": "MEMPHIS GRIZZLIES",
    "Miami Heat": "MIAMI HEAT",
    "Milwaukee Bucks": "MILWAUKEE BUCKS",
    "Minnesota Timberwolves": "MINNESOTA TIMBERWOLVES",
    "New Orleans Pelicans": "NEW ORLEANS PELICANS",
    "New York Knicks": "NEW YORK KNICKS",
    "Oklahoma City Thunder": "OKLAHOMA CITY THUNDER",
    "Orlando Magic": "ORLANDO MAGIC",
    "Philadelphia 76ers": "PHILADELPHIA 76ERS",
    "Phoenix Suns": "PHOENIX SUNS",
    "Portland Trail Blazers": "PORTLAND TRAIL BLAZERS",
    "Sacramento Kings": "SACRAMENTO KINGS",
    "San Antonio Spurs": "SAN ANTONIO SPURS",
    "Toronto Raptors": "TORONTO RAPTORS",
    "Utah Jazz": "UTAH JAZZ",
    "Washington Wizards": "WASHINGTON WIZARDS",
}


def fetch_todays_nba_odds(api_key: str) -> list:
    """Fetch today's NBA games + odds from the-odds-api.com."""
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "bookmakers": "draftkings,fanduel,betmgm,caesars",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    print(f"Odds API requests remaining: {resp.headers.get('x-requests-remaining', '?')}")
    games = resp.json()
    today = date.today().isoformat()
    return [g for g in games if g["commence_time"][:10] == today]


def get_team_stats() -> pd.DataFrame:
    """Load the most recent rolling stats for each NBA team from cached game logs."""
    from data.nba_fetcher import build_team_game_log
    log = build_team_game_log(list(range(2022, 2026)))
    latest = log.sort_values("date").groupby("team").last().reset_index()
    return latest


def build_today_features(games: list, team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Build a matchup feature row for each of today's games.
    Returns a DataFrame aligned to the games list.
    """
    stat_cols = [c for c in team_stats.columns if any(
        c.startswith(p) for p in ["win_pct", "pts_for_avg", "pts_against_avg", "point_diff_avg"]
    )] + ["days_rest"]

    rows = []
    for g in games:
        home_api = g["home_team"]
        away_api = g["away_team"]
        home_br = TEAM_MAP.get(home_api, home_api.upper())
        away_br = TEAM_MAP.get(away_api, away_api.upper())

        home_row = team_stats[team_stats["team"] == home_br]
        away_row = team_stats[team_stats["team"] == away_br]

        if home_row.empty or away_row.empty:
            print(f"  Warning: missing stats for {home_api} or {away_api}, skipping")
            rows.append(None)
            continue

        h = home_row.iloc[0]
        a = away_row.iloc[0]
        row = {"game_id": g["id"], "home_team": home_api, "away_team": away_api}

        for c in stat_cols:
            row[f"home_{c}"] = h.get(c, np.nan)
            row[f"away_{c}"] = a.get(c, np.nan)

        for window in [5, 10, 20]:
            row[f"win_pct_diff_{window}g"] = (
                h.get(f"win_pct_{window}g", np.nan) - a.get(f"win_pct_{window}g", np.nan))
            row[f"point_diff_diff_{window}g"] = (
                h.get(f"point_diff_avg_{window}g", np.nan) - a.get(f"point_diff_avg_{window}g", np.nan))

        row["rest_advantage"] = h.get("days_rest", 0) - a.get("days_rest", 0)
        rows.append(row)

    return rows  # list, some entries may be None


def best_line(game: dict, side: str) -> float | None:
    """Return the best available moneyline for 'home' or 'away' across all bookmakers."""
    best = None
    for book in game.get("bookmakers", []):
        h2h = next((m for m in book["markets"] if m["key"] == "h2h"), None)
        if not h2h:
            continue
        team_name = game["home_team"] if side == "home" else game["away_team"]
        outcome = next((o for o in h2h["outcomes"] if o["name"] == team_name), None)
        if not outcome:
            continue
        price = outcome["price"]
        if best is None:
            best = price
        else:
            # Higher odds = better payout — prefer bigger positive or less negative
            imp_best = 100/(best+100) if best>0 else abs(best)/(abs(best)+100)
            imp_new  = 100/(price+100) if price>0 else abs(price)/(abs(price)+100)
            if imp_new < imp_best:  # lower implied = better odds for bettor
                best = price
    return best


def get_spread(game: dict, side: str) -> dict | None:
    for book in game.get("bookmakers", []):
        spreads = next((m for m in book["markets"] if m["key"] == "spreads"), None)
        if not spreads:
            continue
        team = game["home_team"] if side == "home" else game["away_team"]
        o = next((x for x in spreads["outcomes"] if x["name"] == team), None)
        if o:
            return {"point": o["point"], "price": o["price"]}
    return None


def get_total(game: dict) -> dict | None:
    for book in game.get("bookmakers", []):
        totals = next((m for m in book["markets"] if m["key"] == "totals"), None)
        if not totals:
            continue
        over  = next((x for x in totals["outcomes"] if x["name"] == "Over"), None)
        under = next((x for x in totals["outcomes"] if x["name"] == "Under"), None)
        if over and under:
            return {"line": over["point"], "over": over["price"], "under": under["price"]}
    return None


def run(api_key: str, output_path: str = "predictions.json") -> None:
    print("Fetching today's NBA games...")
    games = fetch_todays_nba_odds(api_key)
    print(f"Found {len(games)} NBA games today")

    if not games:
        result = {"generated_at": datetime.now(timezone.utc).isoformat(), "games": []}
        Path(output_path).write_text(json.dumps(result, indent=2))
        print("No games today — wrote empty predictions.json")
        return

    print("Loading team stats...")
    team_stats = get_team_stats()

    print("Building features...")
    feature_rows = build_today_features(games, team_stats)

    print("Loading model...")
    bundle = load_model(sport="nba", model_type="xgb")

    output_games = []
    for i, game in enumerate(games):
        feat_row = feature_rows[i]
        home_odds = best_line(game, "home")
        away_odds = best_line(game, "away")
        spread    = get_spread(game, "home")
        total     = get_total(game)

        game_out = {
            "id": game["id"],
            "sport": "basketball_nba",
            "sport_label": "NBA",
            "commence_time": game["commence_time"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "home_odds": home_odds,
            "away_odds": away_odds,
            "spread": spread,
            "total": total,
            "model_home_prob": None,
            "model_away_prob": None,
            "home_no_vig_prob": None,
            "away_no_vig_prob": None,
            "home_edge": None,
            "away_edge": None,
            "home_ev": None,
            "away_ev": None,
            "home_kelly": None,
            "away_kelly": None,
            "best_bet": None,
        }

        # No-vig market probabilities
        if home_odds and away_odds:
            home_nv, away_nv = remove_vig(home_odds, away_odds)
            game_out["home_no_vig_prob"] = round(home_nv, 4)
            game_out["away_no_vig_prob"] = round(away_nv, 4)

        # Model predictions
        if feat_row is not None:
            feat_df = pd.DataFrame([feat_row])
            feat_df = feat_df.dropna(axis=1)
            available = [c for c in bundle["features"] if c in feat_df.columns]
            if len(available) >= len(bundle["features"]) * 0.7:
                feat_df_aligned = pd.DataFrame(columns=bundle["features"])
                for c in bundle["features"]:
                    feat_df_aligned[c] = feat_df[c] if c in feat_df.columns else 0.0
                prob_home = predict_proba(bundle, feat_df_aligned)[0]
                prob_away = 1 - prob_home
                game_out["model_home_prob"] = round(float(prob_home), 4)
                game_out["model_away_prob"] = round(float(prob_away), 4)

                if home_odds and away_odds:
                    home_nv = game_out["home_no_vig_prob"]
                    away_nv = game_out["away_no_vig_prob"]
                    h_edge = round(float(prob_home) - home_nv, 4)
                    a_edge = round(float(prob_away) - away_nv, 4)
                    game_out["home_edge"] = h_edge
                    game_out["away_edge"] = a_edge
                    game_out["home_ev"]   = round(expected_value(float(prob_home), home_odds), 4)
                    game_out["away_ev"]   = round(expected_value(float(prob_away), away_odds), 4)
                    game_out["home_kelly"] = round(kelly_calc(float(prob_home), home_odds) * 0.25, 4)
                    game_out["away_kelly"] = round(kelly_calc(float(prob_away), away_odds) * 0.25, 4)

                    # Best bet flag
                    best_edge = max(h_edge, a_edge)
                    if best_edge >= 0.05:
                        side = "home" if h_edge >= a_edge else "away"
                        team = game["home_team"] if side == "home" else game["away_team"]
                        odds = home_odds if side == "home" else away_odds
                        ev   = game_out["home_ev"] if side == "home" else game_out["away_ev"]
                        game_out["best_bet"] = {
                            "side": side,
                            "team": team,
                            "odds": odds,
                            "edge": round(best_edge, 4),
                            "ev": round(ev, 4),
                            "strength": "strong" if best_edge >= 0.05 else "moderate",
                        }
                    elif best_edge >= 0.03:
                        side = "home" if h_edge >= a_edge else "away"
                        team = game["home_team"] if side == "home" else game["away_team"]
                        odds = home_odds if side == "home" else away_odds
                        ev   = game_out["home_ev"] if side == "home" else game_out["away_ev"]
                        game_out["best_bet"] = {
                            "side": side,
                            "team": team,
                            "odds": odds,
                            "edge": round(best_edge, 4),
                            "ev": round(ev, 4),
                            "strength": "moderate",
                        }

        output_games.append(game_out)
        print(f"  {game['away_team']} @ {game['home_team']}: "
              f"model={game_out['model_home_prob']} "
              f"market={game_out['home_no_vig_prob']} "
              f"edge={game_out['home_edge']}")

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date": date.today().isoformat(),
        "games": output_games,
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"\nWrote {len(output_games)} predictions to {output_path}")

    strong = sum(1 for g in output_games if g.get("best_bet") and g["best_bet"]["strength"] == "strong")
    moderate = sum(1 for g in output_games if g.get("best_bet") and g["best_bet"]["strength"] == "moderate")
    print(f"Strong edges: {strong}  |  Moderate edges: {moderate}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("ODDS_API_KEY", ""))
    parser.add_argument("--output", default="predictions.json")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: provide --api-key or set ODDS_API_KEY in .env")
        sys.exit(1)

    run(args.api_key, args.output)
