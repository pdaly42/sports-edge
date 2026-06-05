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
    """
    Fetch all games + odds for a sport/date from the-odds-api.com.
    Returns empty list (instead of raising) on quota exhaustion or auth failure
    so the pipeline can still write model probabilities without odds.
    """
    if not api_key:
        print(f"  {sport_key}: no API key — skipping odds fetch")
        return []

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey":     api_key,
        "regions":    "us",
        "markets":    "h2h,spreads,totals",
        "oddsFormat": "american",
        "bookmakers": "draftkings,fanduel,betmgm,caesars",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
    except Exception as e:
        print(f"  {sport_key}: network error — {e}")
        return []

    if resp.status_code == 401:
        print(f"  {sport_key}: odds API quota exhausted or key invalid (401) — skipping odds")
        return []
    if resp.status_code == 429:
        print(f"  {sport_key}: odds API rate limited (429) — skipping odds")
        return []
    if resp.status_code == 422:
        print(f"  {sport_key}: not in season")
        return []
    if not resp.ok:
        print(f"  {sport_key}: odds API error {resp.status_code} — skipping odds")
        return []

    print(f"  Requests remaining: {resp.headers.get('x-requests-remaining', '?')}")
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


ESPN_SPORT_MAP = {
    "basketball_nba":        "basketball/nba",
    "baseball_mlb":          "baseball/mlb",
    "americanfootball_nfl":  "football/nfl",
    "soccer_fifa_world_cup": "soccer/fifa.world",
}

def fetch_games_from_espn(sport_key: str, target_date: str) -> list:
    """
    Fallback game source when odds API is unavailable.
    Returns minimal game dicts (no bookmakers/odds) using ESPN's free API.
    """
    espn_path = ESPN_SPORT_MAP.get(sport_key)
    if not espn_path:
        return []

    date_str = target_date.replace("-", "")  # YYYYMMDD
    url = f"https://site.api.espn.com/apis/site/v2/sports/{espn_path}/scoreboard?dates={date_str}"
    try:
        resp = requests.get(url, timeout=10)
        if not resp.ok:
            return []
        events = resp.json().get("events", [])
    except Exception:
        return []

    games = []
    for ev in events:
        try:
            comp  = ev["competitions"][0]
            teams = comp["competitors"]
            home  = next((t for t in teams if t.get("homeAway") == "home"), teams[0])
            away  = next((t for t in teams if t.get("homeAway") == "away"), teams[1])
            games.append({
                "id":             ev["id"],
                "commence_time":  ev["date"],
                "home_team":      home["team"]["displayName"],
                "away_team":      away["team"]["displayName"],
                "bookmakers":     [],   # no odds available
            })
        except Exception:
            continue
    print(f"  {sport_key}: loaded {len(games)} games from ESPN (no odds)")
    return games


def build_game_prediction(game: dict, model_home_prob,
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
    if not games:
        games = fetch_games_from_espn("basketball_nba", target_date)
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
    if not games:
        games = fetch_games_from_espn("baseball_mlb", target_date)
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
# Soccer / World Cup  (3-way: home win / draw / away win)
# ─────────────────────────────────────────────────────────────

SOCCER_ODDS_TO_ESPN = {
    "United States": "USA",
    "South Korea": "Korea Republic",
    "Czech Republic": "Czechia",
    "Ivory Coast": "Côte d'Ivoire",
    "Iran": "IR Iran",
    "DR Congo": "DR Congo",
    "Trinidad & Tobago": "Trinidad and Tobago",
    "Bosnia & Herzegovina": "Bosnia-Herzegovina",
}


def no_vig_3way(h2h_market: dict) -> tuple:
    """Remove vig from a 3-outcome soccer market. Returns (home_p, draw_p, away_p)."""
    outcomes = h2h_market.get("outcomes", [])
    if len(outcomes) < 3:
        return None, None, None
    raw = {o["name"]: americanToImpl_py(o["price"]) for o in outcomes}
    total = sum(raw.values())
    return (
        raw.get(list(raw.keys())[0], 0) / total,
        raw.get("Draw", 0) / total,
        raw.get(list(raw.keys())[-1], 0) / total,
    )


def americanToImpl_py(odds: float) -> float:
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)


def get_soccer_h2h(game: dict) -> dict:
    for book in game.get("bookmakers", []):
        m = next((x for x in book["markets"] if x["key"] == "h2h"), None)
        if m and len(m["outcomes"]) == 3:
            return m
    return {}


def predict_soccer(api_key: str, target_date: str) -> list:
    print("\n── Soccer / World Cup ──")
    games = fetch_odds(api_key, "soccer_fifa_world_cup", target_date)
    if not games:
        games = fetch_games_from_espn("soccer_fifa_world_cup", target_date)
    print(f"  {len(games)} WC games on {target_date}")
    if not games:
        return []

    import json
    from models.soccer_trainer import load_soccer_model, predict_soccer_proba
    from data.soccer_fetcher import ODDS_TO_ESPN, get_current_team_form

    # Load ELO ratings
    elo_path = Path("data/raw/soccer/current_elo.json")
    elo = json.loads(elo_path.read_text()) if elo_path.exists() else {}

    # Load team form
    all_teams = [t for g in games for t in [g["home_team"], g["away_team"]]]
    espn_teams = [SOCCER_ODDS_TO_ESPN.get(t, t) for t in all_teams]
    form = get_current_team_form(espn_teams)

    bundle = load_soccer_model(sport="soccer_wc")
    results = []

    for game in games:
        home_api = game["home_team"]
        away_api = game["away_team"]
        home_espn = SOCCER_ODDS_TO_ESPN.get(home_api, home_api)
        away_espn = SOCCER_ODDS_TO_ESPN.get(away_api, away_api)

        home_elo = elo.get(home_espn, elo.get(home_api, 1750))
        away_elo = elo.get(away_espn, elo.get(away_api, 1750))

        home_form = form.get(home_espn, form.get(home_api, {}))
        away_form = form.get(away_espn, form.get(away_api, {}))

        feat = {
            "elo_diff":            home_elo - away_elo,
            "home_elo_pre":        home_elo,
            "away_elo_pre":        away_elo,
            "home_win_pct_10g":    home_form.get("win_pct_10g", 0.45),
            "away_win_pct_10g":    away_form.get("win_pct_10g", 0.45),
            "home_gf_avg_10g":     home_form.get("gf_avg_10g", 1.2),
            "away_gf_avg_10g":     away_form.get("gf_avg_10g", 1.2),
            "home_ga_avg_10g":     home_form.get("ga_avg_10g", 1.2),
            "away_ga_avg_10g":     away_form.get("ga_avg_10g", 1.2),
            "home_gd_avg_10g":     home_form.get("gd_avg_10g", 0.0),
            "away_gd_avg_10g":     away_form.get("gd_avg_10g", 0.0),
            "win_pct_diff_10g":    home_form.get("win_pct_10g", 0.45) - away_form.get("win_pct_10g", 0.45),
            "gd_diff_10g":         home_form.get("gd_avg_10g", 0.0)  - away_form.get("gd_avg_10g", 0.0),
        }

        feat_df = pd.DataFrame([feat])
        probs = predict_soccer_proba(bundle, feat_df)[0]
        # probs: [P(away_win), P(draw), P(home_win)]
        p_away, p_draw, p_home = float(probs[0]), float(probs[1]), float(probs[2])

        # Market no-vig probabilities
        h2h = get_soccer_h2h(game)
        mkt_home, mkt_draw, mkt_away = None, None, None
        home_odds = away_odds = draw_odds = None
        if h2h:
            outcomes = h2h.get("outcomes", [])
            for o in outcomes:
                if o["name"] == home_api:
                    home_odds = o["price"]
                elif o["name"] == away_api:
                    away_odds = o["price"]
                elif o["name"] == "Draw":
                    draw_odds = o["price"]
            if home_odds and away_odds and draw_odds:
                impl = [americanToImpl_py(home_odds),
                        americanToImpl_py(draw_odds),
                        americanToImpl_py(away_odds)]
                tot = sum(impl)
                mkt_home = impl[0] / tot
                mkt_draw = impl[1] / tot
                mkt_away = impl[2] / tot

        # Compute edges and EV for all 3 outcomes
        def edge_ev(model_p, mkt_p, odds):
            if mkt_p is None or odds is None:
                return None, None
            ev = model_p * (abs(odds)/100 if odds < 0 else odds/100) - (1 - model_p)
            return round(model_p - mkt_p, 4), round(ev, 4)

        h_edge, h_ev = edge_ev(p_home, mkt_home, home_odds)
        d_edge, d_ev = edge_ev(p_draw, mkt_draw, draw_odds)
        a_edge, a_ev = edge_ev(p_away, mkt_away, away_odds)

        # Best bet across all 3 outcomes
        best_bet = None
        edges = [
            ("home", p_home, home_odds, h_edge, h_ev, home_api),
            ("draw", p_draw, draw_odds, d_edge, d_ev, "Draw"),
            ("away", p_away, away_odds, a_edge, a_ev, away_api),
        ]
        best = max(edges, key=lambda x: x[3] or -999)
        if best[3] is not None and best[3] >= 0.03:
            best_bet = {
                "side": best[0], "team": best[5], "odds": best[2],
                "edge": best[3], "ev": best[4],
                "strength": "strong" if best[3] >= 0.05 else "moderate",
            }

        pred = {
            "id": game["id"], "sport": "soccer_fifa_world_cup",
            "sport_label": "World Cup",
            "commence_time": game["commence_time"],
            "home_team": home_api, "away_team": away_api,
            "home_odds": home_odds, "away_odds": away_odds, "draw_odds": draw_odds,
            "spread": get_spread(game, "home"), "total": get_total(game),
            "model_home_prob": round(p_home, 4),
            "model_draw_prob": round(p_draw, 4),
            "model_away_prob": round(p_away, 4),
            "home_no_vig_prob": round(mkt_home, 4) if mkt_home else None,
            "draw_no_vig_prob": round(mkt_draw, 4) if mkt_draw else None,
            "away_no_vig_prob": round(mkt_away, 4) if mkt_away else None,
            "home_edge": h_edge, "draw_edge": d_edge, "away_edge": a_edge,
            "home_ev": h_ev, "draw_ev": d_ev, "away_ev": a_ev,
            "home_elo": round(home_elo), "away_elo": round(away_elo),
            "best_bet": best_bet,
        }
        results.append(pred)
        print(f"  {away_api}({round(away_elo)}) @ {home_api}({round(home_elo)}): "
              f"H{p_home:.0%}/D{p_draw:.0%}/A{p_away:.0%} "
              f"best_edge={best[3]}")

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
    if "soccer" in sports:
        all_games += predict_soccer(api_key, target_date)

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
        print("Warning: no ODDS_API_KEY — will use ESPN for game list, no odds data")

    sports = [args.sport] if args.sport else ["nba", "mlb", "soccer"]
    target_date = args.date or date.today().isoformat()
    output_path = args.output or f"predictions_{target_date}.json"

    try:
        run(args.api_key, output_path, target_date, sports)
    except Exception as e:
        print(f"Pipeline error: {e}")
        # Write a minimal valid file so the workflow exits 0 and the dashboard
        # doesn't break — any partial predictions already computed are lost here
        # but at least the Action succeeds.
        import traceback; traceback.print_exc()
        fallback = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "date":         target_date,
            "games":        [],
            "error":        str(e),
        }
        Path(output_path).write_text(json.dumps(fallback, indent=2))
        print(f"Wrote empty fallback to {output_path} — check logs above for root cause")
        sys.exit(0)  # exit 0 so the git commit + push steps still run
