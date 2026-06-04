"""
Daily prediction script — NBA + MLB.
Fetches odds, loads current team stats, runs each sport's trained model,
and writes predictions_YYYY-MM-DD.json for the dashboard.

Usage:
  python3 predict_today.py                    # today, reads ODDS_API_KEY from .env
  python3 predict_today.py --date 2026-06-05  # specific date
  python3 predict_today.py --sport nba        # one sport only
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

# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

def fetch_odds(api_key: str, sport_key: str, target_date: str) -> list:
    """Fetch all games + odds for a sport/date from the-odds-api.com."""
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey":    api_key,
        "regions":   "us",
        "markets":   "h2h,spreads,totals",
        "oddsFormat": "american",
        "bookmakers": "draftkings,fanduel,betmgm,caesars",
    }
    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code == 422:
        print(f"  {sport_key}: not in season")
        return []
    resp.raise_for_status()
    print(f"  Requests remaining: {resp.headers.get('x-requests-remaining', '?')}")
    # Filter to target date in local time
    all_games = resp.json()
    return [g for g in all_games if g["commence_time"][:10] == target_date]


def best_line(game: dict, side: str) -> float | None:
    """Best available moneyline across all bookmakers for a side."""
    best = None
    for book in game.get("bookmakers", []):
        h2h = next((m for m in book["markets"] if m["key"] == "h2h"), None)
        if not h2h:
            continue
        team = game["home_team"] if side == "home" else game["away_team"]
        o = next((x for x in h2h["outcomes"] if x["name"] == team), None)
        if not o:
            continue
        p = o["price"]
        if best is None:
            best = p
        else:
            impl_best = 100/(best+100) if best > 0 else abs(best)/(abs(best)+100)
            impl_new  = 100/(p+100)    if p    > 0 else abs(p)   /(abs(p)   +100)
            if impl_new < impl_best:
                best = p
    return best


def get_spread(game: dict, side: str) -> dict | None:
    for book in game.get("bookmakers", []):
        m = next((x for x in book["markets"] if x["key"] == "spreads"), None)
        if not m:
            continue
        team = game["home_team"] if side == "home" else game["away_team"]
        o = next((x for x in m["outcomes"] if x["name"] == team), None)
        if o:
            return {"point": o["point"], "price": o["price"]}
    return None


def get_total(game: dict) -> dict | None:
    for book in game.get("bookmakers", []):
        m = next((x for x in book["markets"] if x["key"] == "totals"), None)
        if not m:
            continue
        over  = next((x for x in m["outcomes"] if x["name"] == "Over"),  None)
        under = next((x for x in m["outcomes"] if x["name"] == "Under"), None)
        if over and under:
            return {"line": over["point"], "over": over["price"], "under": under["price"]}
    return None


def build_game_prediction(game: dict, model_home_prob: float | None,
                           home_odds: float | None, away_odds: float | None,
                           sport: str, sport_label: str) -> dict:
    """Assemble the full prediction dict for one game."""
    out = {
        "id":             game["id"],
        "sport":          sport,
        "sport_label":    sport_label,
        "commence_time":  game["commence_time"],
        "home_team":      game["home_team"],
        "away_team":      game["away_team"],
        "home_odds":      home_odds,
        "away_odds":      away_odds,
        "spread":         get_spread(game, "home"),
        "total":          get_total(game),
        "model_home_prob": None,
        "model_away_prob": None,
        "home_no_vig_prob": None,
        "away_no_vig_prob": None,
        "home_edge":  None, "away_edge":  None,
        "home_ev":    None, "away_ev":    None,
        "home_kelly": None, "away_kelly": None,
        "best_bet":   None,
    }

    if home_odds and away_odds:
        h_nv, a_nv = remove_vig(home_odds, away_odds)
        out["home_no_vig_prob"] = round(h_nv, 4)
        out["away_no_vig_prob"] = round(a_nv, 4)

        if model_home_prob is not None:
            prob_h = float(model_home_prob)
            prob_a = 1 - prob_h
            out["model_home_prob"] = round(prob_h, 4)
            out["model_away_prob"] = round(prob_a, 4)

            h_edge = round(prob_h - h_nv, 4)
            a_edge = round(prob_a - a_nv, 4)
            out["home_edge"]  = h_edge
            out["away_edge"]  = a_edge
            out["home_ev"]    = round(expected_value(prob_h, home_odds), 4)
            out["away_ev"]    = round(expected_value(prob_a, away_odds), 4)
            out["home_kelly"] = round(kelly_calc(prob_h, home_odds) * 0.25, 4)
            out["away_kelly"] = round(kelly_calc(prob_a, away_odds) * 0.25, 4)

            best_edge = max(h_edge, a_edge)
            if best_edge >= 0.03:
                side = "home" if h_edge >= a_edge else "away"
                out["best_bet"] = {
                    "side":     side,
                    "team":     game["home_team"] if side == "home" else game["away_team"],
                    "odds":     home_odds if side == "home" else away_odds,
                    "edge":     round(best_edge, 4),
                    "ev":       out["home_ev"] if side == "home" else out["away_ev"],
                    "strength": "strong" if best_edge >= 0.05 else "moderate",
                }
    return out


def align_features(feat_row: dict, bundle: dict) -> pd.DataFrame | None:
    """Align a raw feature dict to the model's expected feature list."""
    df = pd.DataFrame([feat_row])
    available = [c for c in bundle["features"] if c in df.columns]
    if len(available) < len(bundle["features"]) * 0.7:
        return None
    aligned = pd.DataFrame(columns=bundle["features"])
    for c in bundle["features"]:
        aligned[c] = df[c].values if c in df.columns else 0.0
    return aligned


# ─────────────────────────────────────────────────────────────
# NBA
# ─────────────────────────────────────────────────────────────

NBA_TEAM_MAP = {
    "Atlanta Hawks": "ATLANTA HAWKS", "Boston Celtics": "BOSTON CELTICS",
    "Brooklyn Nets": "BROOKLYN NETS", "Charlotte Hornets": "CHARLOTTE HORNETS",
    "Chicago Bulls": "CHICAGO BULLS", "Cleveland Cavaliers": "CLEVELAND CAVALIERS",
    "Dallas Mavericks": "DALLAS MAVERICKS", "Denver Nuggets": "DENVER NUGGETS",
    "Detroit Pistons": "DETROIT PISTONS", "Golden State Warriors": "GOLDEN STATE WARRIORS",
    "Houston Rockets": "HOUSTON ROCKETS", "Indiana Pacers": "INDIANA PACERS",
    "Los Angeles Clippers": "LOS ANGELES CLIPPERS", "Los Angeles Lakers": "LOS ANGELES LAKERS",
    "Memphis Grizzlies": "MEMPHIS GRIZZLIES", "Miami Heat": "MIAMI HEAT",
    "Milwaukee Bucks": "MILWAUKEE BUCKS", "Minnesota Timberwolves": "MINNESOTA TIMBERWOLVES",
    "New Orleans Pelicans": "NEW ORLEANS PELICANS", "New York Knicks": "NEW YORK KNICKS",
    "Oklahoma City Thunder": "OKLAHOMA CITY THUNDER", "Orlando Magic": "ORLANDO MAGIC",
    "Philadelphia 76ers": "PHILADELPHIA 76ERS", "Phoenix Suns": "PHOENIX SUNS",
    "Portland Trail Blazers": "PORTLAND TRAIL BLAZERS", "Sacramento Kings": "SACRAMENTO KINGS",
    "San Antonio Spurs": "SAN ANTONIO SPURS", "Toronto Raptors": "TORONTO RAPTORS",
    "Utah Jazz": "UTAH JAZZ", "Washington Wizards": "WASHINGTON WIZARDS",
}


def predict_nba(api_key: str, target_date: str) -> list:
    print("\n── NBA ──")
    games = fetch_odds(api_key, "basketball_nba", target_date)
    print(f"  {len(games)} games on {target_date}")
    if not games:
        return []

    from data.nba_fetcher import build_team_game_log
    stats = build_team_game_log(list(range(2022, 2026)))
    latest = stats.sort_values("date").groupby("team").last().reset_index()

    stat_cols = [c for c in latest.columns if any(
        c.startswith(p) for p in ["win_pct", "pts_for_avg", "pts_against_avg", "point_diff_avg"]
    )] + ["days_rest"]

    bundle = load_model(sport="nba", model_type="xgb")
    results = []

    for game in games:
        home_api = game["home_team"]
        away_api = game["away_team"]
        home_br  = NBA_TEAM_MAP.get(home_api, home_api.upper())
        away_br  = NBA_TEAM_MAP.get(away_api, away_api.upper())

        h_row = latest[latest["team"] == home_br]
        a_row = latest[latest["team"] == away_br]

        prob_home = None
        if not h_row.empty and not a_row.empty:
            h, a = h_row.iloc[0], a_row.iloc[0]
            feat = {"game_id": game["id"]}
            for c in stat_cols:
                feat[f"home_{c}"] = h.get(c, np.nan)
                feat[f"away_{c}"] = a.get(c, np.nan)
            for w in [5, 10, 20]:
                feat[f"win_pct_diff_{w}g"]   = h.get(f"win_pct_{w}g", np.nan)   - a.get(f"win_pct_{w}g", np.nan)
                feat[f"point_diff_diff_{w}g"] = h.get(f"point_diff_avg_{w}g", np.nan) - a.get(f"point_diff_avg_{w}g", np.nan)
            feat["rest_advantage"] = h.get("days_rest", 0) - a.get("days_rest", 0)

            aligned = align_features(feat, bundle)
            if aligned is not None:
                prob_home = predict_proba(bundle, aligned)[0]
        else:
            print(f"  Missing stats: {home_api} or {away_api}")

        pred = build_game_prediction(
            game, prob_home,
            best_line(game, "home"), best_line(game, "away"),
            "basketball_nba", "NBA"
        )
        results.append(pred)
        print(f"  {away_api} @ {home_api}: model={pred['model_home_prob']} edge={pred['home_edge']}")

    return results


# ─────────────────────────────────────────────────────────────
# MLB
# ─────────────────────────────────────────────────────────────

MLB_TEAM_MAP = {
    "Atlanta Braves":        "ATL", "Miami Marlins":         "MIA",
    "New York Mets":         "NYM", "Philadelphia Phillies": "PHI",
    "Washington Nationals":  "WSN", "Chicago Cubs":          "CHC",
    "Cincinnati Reds":       "CIN", "Milwaukee Brewers":     "MIL",
    "Pittsburgh Pirates":    "PIT", "St. Louis Cardinals":   "STL",
    "Arizona Diamondbacks":  "ARI", "Colorado Rockies":      "COL",
    "Los Angeles Dodgers":   "LAD", "San Diego Padres":      "SDP",
    "San Francisco Giants":  "SFG", "Baltimore Orioles":     "BAL",
    "Boston Red Sox":        "BOS", "New York Yankees":      "NYY",
    "Tampa Bay Rays":        "TBR", "Toronto Blue Jays":     "TOR",
    "Chicago White Sox":     "CHW", "Cleveland Guardians":   "CLE",
    "Detroit Tigers":        "DET", "Kansas City Royals":    "KCR",
    "Minnesota Twins":       "MIN", "Houston Astros":        "HOU",
    "Los Angeles Angels":    "LAA", "Oakland Athletics":     "OAK",
    "Seattle Mariners":      "SEA", "Texas Rangers":         "TEX",
}


def predict_mlb(api_key: str, target_date: str) -> list:
    print("\n── MLB ──")
    games = fetch_odds(api_key, "baseball_mlb", target_date)
    print(f"  {len(games)} games on {target_date}")
    if not games:
        return []

    from data.mlb_fetcher import build_team_game_log
    from data.mlb_pitcher_stats import get_probable_pitcher_stats, pitcher_stats_or_median

    # Team rolling stats
    stats = build_team_game_log(list(range(2023, 2026)))
    latest = stats.sort_values("date").groupby("team").last().reset_index()

    stat_cols = [c for c in latest.columns if any(
        c.startswith(p) for p in ["win_pct", "runs_for_avg", "runs_against_avg", "run_diff_avg"]
    )] + ["days_rest"]

    # Today's probable pitchers from MLB Stats API (free, no key)
    print("  Fetching probable pitchers from MLB Stats API...")
    pitcher_stats = get_probable_pitcher_stats(target_date)

    bundle = load_model(sport="mlb", model_type="xgb")
    results = []

    for game in games:
        home_api = game["home_team"]
        away_api = game["away_team"]
        home_br  = MLB_TEAM_MAP.get(home_api)
        away_br  = MLB_TEAM_MAP.get(away_api)

        prob_home = None
        if home_br and away_br:
            h_row = latest[latest["team"] == home_br]
            a_row = latest[latest["team"] == away_br]

            if not h_row.empty and not a_row.empty:
                h, a = h_row.iloc[0], a_row.iloc[0]
                feat = {"game_id": game["id"]}

                # Team rolling stats
                for c in stat_cols:
                    feat[f"home_{c}"] = h.get(c, np.nan)
                    feat[f"away_{c}"] = a.get(c, np.nan)
                for w in [5, 10, 20]:
                    feat[f"win_pct_diff_{w}g"] = h.get(f"win_pct_{w}g", np.nan) - a.get(f"win_pct_{w}g", np.nan)
                    feat[f"run_diff_diff_{w}g"] = h.get(f"run_diff_avg_{w}g", np.nan) - a.get(f"run_diff_avg_{w}g", np.nan)
                feat["rest_advantage"] = h.get("days_rest", 0) - a.get("days_rest", 0)

                # Starting pitcher stats
                home_p = pitcher_stats_or_median(pitcher_stats.get(home_api))
                away_p = pitcher_stats_or_median(pitcher_stats.get(away_api))
                for col in ["era", "whip", "k_per_9", "k_bb_ratio", "innings_pitched"]:
                    feat[f"home_starter_{col}"] = home_p[col]
                    feat[f"away_starter_{col}"] = away_p[col]
                feat["starter_era_diff"]  = home_p["era"]    - away_p["era"]
                feat["starter_whip_diff"] = home_p["whip"]   - away_p["whip"]
                feat["starter_k9_diff"]   = home_p["k_per_9"]- away_p["k_per_9"]

                aligned = align_features(feat, bundle)
                if aligned is not None:
                    prob_home = predict_proba(bundle, aligned)[0]
            else:
                print(f"  Missing team stats: {home_api} or {away_api}")
        else:
            print(f"  Unknown team mapping: {home_api} or {away_api}")

        # Add pitcher names to prediction output for display in dashboard
        home_pitcher_name = pitcher_stats.get(home_api, {}).get("name", "TBD")
        away_pitcher_name = pitcher_stats.get(away_api, {}).get("name", "TBD")

        pred = build_game_prediction(
            game, prob_home,
            best_line(game, "home"), best_line(game, "away"),
            "baseball_mlb", "MLB"
        )
        pred["home_pitcher"] = home_pitcher_name
        pred["away_pitcher"] = away_pitcher_name
        results.append(pred)
        print(f"  {away_api}({away_pitcher_name}) @ {home_api}({home_pitcher_name}): "
              f"model={pred['model_home_prob']} edge={pred['home_edge']}")

    return results


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run(api_key: str, output_path: str = None, target_date: str = None,
        sports: list = None) -> None:
    target_date = target_date or date.today().isoformat()
    output_path = output_path or f"predictions_{target_date}.json"
    sports = sports or ["nba", "mlb"]

    print(f"Running predictions for {target_date} — sports: {sports}")
    all_games = []

    if "nba" in sports:
        all_games += predict_nba(api_key, target_date)
    if "mlb" in sports:
        all_games += predict_mlb(api_key, target_date)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date":         target_date,
        "games":        all_games,
    }
    Path(output_path).write_text(json.dumps(result, indent=2))

    strong   = sum(1 for g in all_games if g.get("best_bet") and g["best_bet"]["strength"] == "strong")
    moderate = sum(1 for g in all_games if g.get("best_bet") and g["best_bet"]["strength"] == "moderate")
    print(f"\nWrote {len(all_games)} predictions to {output_path}")
    print(f"Strong edges: {strong}  |  Moderate edges: {moderate}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("ODDS_API_KEY", ""))
    parser.add_argument("--date",    default=None,  help="YYYY-MM-DD (defaults to today)")
    parser.add_argument("--output",  default=None,  help="Output file (defaults to predictions_YYYY-MM-DD.json)")
    parser.add_argument("--sport",   default=None,  help="nba or mlb (defaults to both)")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: provide --api-key or set ODDS_API_KEY in .env")
        sys.exit(1)

    sports = [args.sport] if args.sport else ["nba", "mlb"]
    target_date = args.date or date.today().isoformat()
    output_path = args.output or f"predictions_{target_date}.json"
    run(args.api_key, output_path, target_date, sports)
