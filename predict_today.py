"""
Daily prediction script — NBA + MLB.
Fetches odds, loads current team stats, runs each sport's trained model,
and writes predictions_YYYY-MM-DD.json for the dashboard.

Usage:
  python3 predict_today.py                    # today, reads ODDS_API_KEY from .env
  python3 predict_today.py --date 2026-06-05  # specific date
  python3 predict_today.py --sport nba        # one sport only
"""

from __future__ import annotations
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
from data.bovada_odds import fetch_bovada_odds
from utils.odds import remove_vig, expected_value, kelly_fraction as kelly_calc

# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

def fetch_odds(api_key: str, sport_key: str, target_date: str) -> list:
    """
    Fetch all games + odds for a sport/date.
    Tries the-odds-api.com first; falls back to Bovada (free, no key) on quota
    exhaustion, auth failure, or missing key.
    """
    if api_key:
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
            resp = None

        if resp is not None and resp.ok:
            print(f"  Requests remaining: {resp.headers.get('x-requests-remaining', '?')}")
            all_games = resp.json()
            return [g for g in all_games if g["commence_time"][:10] == target_date]

        if resp is not None:
            code = resp.status_code
            if code == 422:
                print(f"  {sport_key}: not in season")
                return []
            print(f"  {sport_key}: odds API returned {code} — falling back to Bovada")
    else:
        print(f"  {sport_key}: no API key — using Bovada (free)")

    games = fetch_bovada_odds(sport_key, target_date)
    if not games:
        print(f"  {sport_key}: Bovada returned no games for {target_date}")
    return games


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


def fetch_mlb_team_records() -> dict:
    """
    Fetch live MLB standings from the MLB Stats API.
    Returns dict keyed by full team name → {wins, losses, last10_w, last10_l}.
    """
    url = "https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season=2026&standingsTypes=regularSeason"
    try:
        r = requests.get(url, timeout=10)
        if not r.ok:
            return {}
        records = {}
        for division in r.json().get("records", []):
            for t in division.get("teamRecords", []):
                name = t["team"]["name"]
                last10 = next(
                    (s for s in t.get("records", {}).get("splitRecords", []) if s["type"] == "lastTen"),
                    None
                )
                records[name] = {
                    "wins":     t["wins"],
                    "losses":   t["losses"],
                    "last10_w": last10["wins"]   if last10 else None,
                    "last10_l": last10["losses"] if last10 else None,
                }
        print(f"  Fetched live records for {len(records)} MLB teams")
        return records
    except Exception as e:
        print(f"  Could not fetch MLB standings: {e}")
        return {}


def generate_justification(home: str, away: str, feat: dict,
                            model_home_prob: float, best_bet: dict,
                            sport: str, extra: dict = None) -> str:
    """
    Generate a 1-2 sentence human-readable justification for why the model
    sees an edge. Pulls the most important feature drivers from feat dict.
    Only called when best_bet is not None (i.e., edge >= 3%).
    extra: sport-specific data (e.g. pitcher names/ERA for MLB, ELO for soccer)
    """
    if best_bet is None:
        return ""

    side      = best_bet["side"]
    edge_pct  = round(best_bet["edge"] * 100, 1)
    fav_team  = home if side == "home" else away
    dog_team  = away if side == "home" else home
    prefix    = "home" if side == "home" else "away"
    opp_pfx   = "away" if side == "home" else "home"

    parts = []

    if sport in ("basketball_nba", "baseball_mlb"):

        # Live recent form from standings API
        if extra:
            team_records = extra.get("team_records", {})
            # Standings API uses short names (e.g. "Royals"); match by substring
            def _lookup_rec(full_name):
                if full_name in team_records:
                    return team_records[full_name]
                for short, rec in team_records.items():
                    if short in full_name:
                        return rec
                return None
            fav_rec = _lookup_rec(fav_team)
            dog_rec = _lookup_rec(dog_team)
            if fav_rec and dog_rec:
                fav_l10_w = fav_rec.get("last10_w")
                fav_l10_l = fav_rec.get("last10_l")
                dog_l10_w = dog_rec.get("last10_w")
                dog_l10_l = dog_rec.get("last10_l")
                fav_season = f"{fav_rec['wins']}-{fav_rec['losses']}"
                dog_season = f"{dog_rec['wins']}-{dog_rec['losses']}"
                if fav_l10_w is not None and dog_l10_w is not None:
                    parts.append(
                        f"{fav_team} are {fav_l10_w}-{fav_l10_l} in their last 10 "
                        f"({dog_team} {dog_l10_w}-{dog_l10_l}); "
                        f"season records {fav_season} vs {dog_season}"
                    )

        # Run/point differential
        diff_key = f"{prefix}_point_diff_avg_10g" if sport == "basketball_nba" else f"{prefix}_run_diff_avg_10g"
        opp_diff_key = f"{opp_pfx}_point_diff_avg_10g" if sport == "basketball_nba" else f"{opp_pfx}_run_diff_avg_10g"
        d_fav = feat.get(diff_key)
        d_dog = feat.get(opp_diff_key)
        unit  = "point" if sport == "basketball_nba" else "run"
        if d_fav is not None and str(d_fav) != "nan" and d_dog is not None:
            parts.append(f"{fav_team}'s avg {unit} differential is {d_fav:+.1f} vs {dog_team}'s {d_dog:+.1f} over the last 10 games")

        # Rest advantage (if meaningful)
        rest = feat.get("rest_advantage", 0) or 0
        if side == "away":
            rest = -rest
        if abs(rest) >= 1:
            rested = fav_team if rest > 0 else dog_team
            tired  = dog_team if rest > 0 else fav_team
            parts.append(f"{rested} have a {abs(int(rest))}-day rest edge over {tired}")

        # MLB pitcher matchup
        if sport == "baseball_mlb" and extra:
            h_pitcher = extra.get("home_pitcher")
            a_pitcher = extra.get("away_pitcher")
            h_era     = extra.get("home_era")
            a_era     = extra.get("away_era")
            fav_p     = h_pitcher if side == "home" else a_pitcher
            dog_p     = a_pitcher if side == "home" else h_pitcher
            fav_era   = h_era     if side == "home" else a_era
            dog_era   = a_era     if side == "home" else h_era
            if fav_p and dog_p and fav_era and dog_era:
                era_diff = round(dog_era - fav_era, 2)
                if era_diff >= 0.5:
                    parts.append(f"{fav_p} (ERA {fav_era}) has a meaningful ERA advantage over {dog_p} (ERA {dog_era})")

    elif sport == "soccer_fifa_world_cup":
        # ELO advantage
        h_elo = feat.get("home_elo_pre")
        a_elo = feat.get("away_elo_pre")
        fav_elo = h_elo if side == "home" else a_elo
        dog_elo = a_elo if side == "home" else h_elo
        if fav_elo and dog_elo:
            elo_gap = round(fav_elo - dog_elo)
            if elo_gap > 30:
                parts.append(f"{fav_team} hold a {elo_gap}-point ELO advantage ({round(fav_elo)} vs {round(dog_elo)})")

        # Recent form
        wp_fav = feat.get(f"{'home' if side=='home' else 'away'}_win_pct_10g")
        wp_dog = feat.get(f"{'away' if side=='home' else 'home'}_win_pct_10g")
        if wp_fav is not None and str(wp_fav) != "nan":
            w_fav = round(wp_fav * 10)
            parts.append(f"{fav_team} have won {w_fav} of their last 10 internationals")

    # Market vs model summary
    mkt_implied = round((1 - best_bet["edge"] - best_bet.get("ev", 0) / (1 / best_bet["edge"] - 1 if best_bet["edge"] < 1 else 1)) * 100, 1) if parts else None
    parts.append(f"Model prices {fav_team} at {round(model_home_prob*100 if side=='home' else (1-model_home_prob)*100, 1)}% — a +{edge_pct}% edge over the market line")

    return ". ".join(parts[:3]) + "."


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

    # Always write model probabilities when available — independent of odds
    if model_home_prob is not None:
        prob_h = float(model_home_prob)
        prob_a = 1 - prob_h
        out["model_home_prob"] = round(prob_h, 4)
        out["model_away_prob"] = round(prob_a, 4)

    # Edge/EV/Kelly only computable when we have market odds to compare against
    if home_odds and away_odds:
        h_nv, a_nv = remove_vig(home_odds, away_odds)
        out["home_no_vig_prob"] = round(h_nv, 4)
        out["away_no_vig_prob"] = round(a_nv, 4)

        if model_home_prob is not None:
            prob_h = float(out["model_home_prob"])
            prob_a = float(out["model_away_prob"])

            h_edge = round(prob_h - h_nv, 4)
            a_edge = round(prob_a - a_nv, 4)
            out["home_edge"]  = h_edge
            out["away_edge"]  = a_edge
            out["home_ev"]    = round(expected_value(prob_h, home_odds), 4)
            out["away_ev"]    = round(expected_value(prob_a, away_odds), 4)
            out["home_kelly"] = round(kelly_calc(prob_h, home_odds) * 0.25, 4)
            out["away_kelly"] = round(kelly_calc(prob_a, away_odds) * 0.25, 4)

            best_edge = max(h_edge, a_edge)
            side = "home" if h_edge >= a_edge else "away"
            side_prob = prob_h if side == "home" else prob_a
            # Require meaningful edge AND cap model confidence to guard against overfit
            if best_edge >= 0.08 and side_prob <= 0.70:
                out["best_bet"] = {
                    "side":     side,
                    "team":     game["home_team"] if side == "home" else game["away_team"],
                    "odds":     home_odds if side == "home" else away_odds,
                    "edge":     round(best_edge, 4),
                    "ev":       out["home_ev"] if side == "home" else out["away_ev"],
                    "strength": "strong" if best_edge >= 0.12 else "moderate",
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
        if pred.get("best_bet") and feat:
            pred["justification"] = generate_justification(
                home_api, away_api, feat, float(prob_home),
                pred["best_bet"], "basketball_nba"
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
    "Athletics":             "OAK",
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
        c.startswith(p) for p in ["win_pct", "runs_for_avg", "runs_against_avg", "run_diff_avg",
                                   "season_win_pct", "season_run_diff_avg"]
    )] + ["days_rest"]

    # Today's probable pitchers from MLB Stats API (free, no key)
    print("  Fetching probable pitchers from MLB Stats API...")
    pitcher_stats = get_probable_pitcher_stats(target_date)

    # Live team records and last-10 from MLB Stats API
    team_records = fetch_mlb_team_records()

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
                feat["season_win_pct_diff"]      = h.get("season_win_pct", 0.5) - a.get("season_win_pct", 0.5)
                feat["season_run_diff_avg_diff"] = h.get("season_run_diff_avg", 0.0) - a.get("season_run_diff_avg", 0.0)

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
        if pred.get("best_bet") and feat:
            home_era = pitcher_stats.get(home_api, {}).get("era")
            away_era = pitcher_stats.get(away_api, {}).get("era")
            pred["justification"] = generate_justification(
                home_api, away_api, feat, float(prob_home),
                pred["best_bet"], "baseball_mlb",
                extra={
                    "home_pitcher":  home_pitcher_name, "away_pitcher": away_pitcher_name,
                    "home_era":      home_era,          "away_era":     away_era,
                    "team_records":  team_records,
                }
            )
        results.append(pred)
        print(f"  {away_api}({away_pitcher_name}) @ {home_api}({home_pitcher_name}): "
              f"model={pred['model_home_prob']} edge={pred['home_edge']}")

    return results


# ─────────────────────────────────────────────────────────────
# NFL
# ─────────────────────────────────────────────────────────────

# nfl_data_py uses 2-3 letter abbreviations; map from the-odds-api full names
NFL_TEAM_MAP = {
    "Arizona Cardinals":   "ARI", "Atlanta Falcons":      "ATL",
    "Baltimore Ravens":    "BAL", "Buffalo Bills":        "BUF",
    "Carolina Panthers":   "CAR", "Chicago Bears":        "CHI",
    "Cincinnati Bengals":  "CIN", "Cleveland Browns":     "CLE",
    "Dallas Cowboys":      "DAL", "Denver Broncos":       "DEN",
    "Detroit Lions":       "DET", "Green Bay Packers":    "GB",
    "Houston Texans":      "HOU", "Indianapolis Colts":   "IND",
    "Jacksonville Jaguars":"JAX", "Kansas City Chiefs":   "KC",
    "Las Vegas Raiders":   "LV",  "Los Angeles Chargers": "LAC",
    "Los Angeles Rams":    "LA",  "Miami Dolphins":       "MIA",
    "Minnesota Vikings":   "MIN", "New England Patriots": "NE",
    "New Orleans Saints":  "NO",  "New York Giants":      "NYG",
    "New York Jets":       "NYJ", "Philadelphia Eagles":  "PHI",
    "Pittsburgh Steelers": "PIT", "San Francisco 49ers":  "SF",
    "Seattle Seahawks":    "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans":    "TEN", "Washington Commanders":"WAS",
}


def _best_ou_bet(line: float, predicted: float, p_over: float, p_under: float,
                 over_odds, under_odds, edge_over, edge_under) -> dict | None:
    """Flag an O/U best bet when edge >= 4% and predicted total is >= 3 pts from line."""
    if abs(predicted - line) < 3.0:
        return None
    candidates = []
    if edge_over  is not None and edge_over  >= 0.04:
        candidates.append(("over",  p_over,  over_odds,  edge_over))
    if edge_under is not None and edge_under >= 0.04:
        candidates.append(("under", p_under, under_odds, edge_under))
    if not candidates:
        return None
    side, prob, odds, edge = max(candidates, key=lambda x: x[3])
    return {
        "side":     side,
        "odds":     odds,
        "edge":     round(edge, 4),
        "strength": "strong" if edge >= 0.07 else "moderate",
    }


def _get_nfl_current_week(target_date: str) -> tuple[int, int]:
    """
    Return (season, week) for a given date by loading the NFL schedule.
    Falls back to current calendar year and week 1 if not determinable.
    """
    try:
        import nfl_data_py as nfl
        target_dt = pd.to_datetime(target_date)
        current_year = target_dt.year
        # NFL season year: regular season runs Sep–Jan, so Jan belongs to prior season
        season = current_year if target_dt.month >= 8 else current_year - 1
        sched = nfl.import_schedules([season])
        sched = sched[sched["game_type"] == "REG"].copy()
        sched["gameday"] = pd.to_datetime(sched["gameday"])
        # Find the week whose games are closest to target_date
        week_dates = sched.groupby("week")["gameday"].min().reset_index()
        week_dates["delta"] = (week_dates["gameday"] - target_dt).abs()
        week = int(week_dates.loc[week_dates["delta"].idxmin(), "week"])
        return season, week
    except Exception:
        return pd.Timestamp.now().year, 1


def predict_nfl(api_key: str, target_date: str) -> list:
    print("\n── NFL ──")
    games = fetch_odds(api_key, "americanfootball_nfl", target_date)
    if not games:
        games = fetch_games_from_espn("americanfootball_nfl", target_date)
    print(f"  {len(games)} games on {target_date}")
    if not games:
        return []

    from data.nfl_fetcher import get_current_team_stats
    from data.nfl_qb_stats import get_current_qb_stats
    from data.nfl_injuries import get_current_injury_scores

    try:
        bundle = load_model(sport="nfl", model_type="xgb")
    except FileNotFoundError:
        print("  NFL model not found — run scripts/train_if_needed.py first")
        return []

    try:
        totals_bundle = load_model(sport="nfl", model_type="totals")
    except FileNotFoundError:
        totals_bundle = None
        print("  NFL totals model not found — over/under predictions disabled")

    # Team rolling stats
    stats = get_current_team_stats()
    roll_cols = [c for c in stats.columns if any(
        c.startswith(p) for p in
        ["win_pct_", "pts_for_avg_", "pts_against_avg_", "point_diff_avg_",
         "season_win_pct", "season_point_diff_avg"]
    )]

    # Current QB stats
    print("  Fetching current QB stats...")
    qb_stats = get_current_qb_stats()
    qb_roll_cols = [c for c in qb_stats.columns if any(
        c.startswith(p) for p in ["qb_epa_per_att_", "qb_comp_pct_", "qb_ypa_", "qb_td_int_ratio_"]
    )]

    # Current injury report
    season, week = _get_nfl_current_week(target_date)
    print(f"  Fetching injury report for {season} week {week}...")
    try:
        injury_scores = get_current_injury_scores(season, week)
    except Exception as e:
        print(f"  Injury data unavailable (offseason or error): {e}")
        injury_scores = {}

    results = []
    for game in games:
        home_api  = game["home_team"]
        away_api  = game["away_team"]
        home_abbr = NFL_TEAM_MAP.get(home_api)
        away_abbr = NFL_TEAM_MAP.get(away_api)

        prob_home = None
        feat = {}
        if home_abbr and away_abbr:
            h_row = stats[stats["team"] == home_abbr]
            a_row = stats[stats["team"] == away_abbr]

            if not h_row.empty and not a_row.empty:
                h, a = h_row.iloc[0], a_row.iloc[0]
                feat = {"game_id": game["id"]}

                # Team rolling stats
                for c in roll_cols:
                    feat[f"home_{c}"] = h.get(c, np.nan)
                    feat[f"away_{c}"] = a.get(c, np.nan)
                for window in [4, 8, 16]:
                    feat[f"win_pct_diff_{window}g"] = (
                        h.get(f"win_pct_{window}g", np.nan) - a.get(f"win_pct_{window}g", np.nan)
                    )
                    feat[f"point_diff_diff_{window}g"] = (
                        h.get(f"point_diff_avg_{window}g", np.nan) - a.get(f"point_diff_avg_{window}g", np.nan)
                    )
                feat["rest_advantage"]             = h.get("rest", 7)  - a.get("rest", 7)
                feat["home_rest"]                  = h.get("rest", 7)
                feat["away_rest"]                  = a.get("rest", 7)
                feat["season_win_pct_diff"]         = h.get("season_win_pct", 0.5) - a.get("season_win_pct", 0.5)
                feat["season_point_diff_avg_diff"]  = h.get("season_point_diff_avg", 0.0) - a.get("season_point_diff_avg", 0.0)

                # QB rolling stats
                h_qb = qb_stats[qb_stats["team"] == home_abbr]
                a_qb = qb_stats[qb_stats["team"] == away_abbr]
                hq = h_qb.iloc[0] if not h_qb.empty else {}
                aq = a_qb.iloc[0] if not a_qb.empty else {}
                for c in qb_roll_cols:
                    feat[f"home_{c}"] = hq.get(c, 0.0) if hasattr(hq, "get") else 0.0
                    feat[f"away_{c}"] = aq.get(c, 0.0) if hasattr(aq, "get") else 0.0
                for c in qb_roll_cols:
                    feat[f"qb_{c.replace('qb_','')}_diff"] = feat[f"home_{c}"] - feat[f"away_{c}"]

                # Injury scores (0 if team not on report = healthy)
                h_inj = injury_scores.get(home_abbr, {"injury_score": 0.0, "qb_injury_impact": 0.0})
                a_inj = injury_scores.get(away_abbr, {"injury_score": 0.0, "qb_injury_impact": 0.0})
                feat["home_injury_score"]      = h_inj["injury_score"]
                feat["away_injury_score"]      = a_inj["injury_score"]
                feat["home_qb_injury_impact"]  = h_inj["qb_injury_impact"]
                feat["away_qb_injury_impact"]  = a_inj["qb_injury_impact"]
                feat["injury_score_diff"]      = a_inj["injury_score"]     - h_inj["injury_score"]
                feat["qb_injury_impact_diff"]  = a_inj["qb_injury_impact"] - h_inj["qb_injury_impact"]

                # Totals features — combined (sum) versions of scoring/QB/injury
                h_stats = h  # alias for readability
                a_stats = a
                for window in [4, 8, 16]:
                    for stat in ["pts_for_avg", "pts_against_avg", "point_diff_avg"]:
                        hv = h_stats.get(f"{stat}_{window}g", np.nan)
                        av = a_stats.get(f"{stat}_{window}g", np.nan)
                        feat[f"home_{stat}_{window}g"] = hv
                        feat[f"away_{stat}_{window}g"] = av
                        if stat in ("pts_for_avg", "pts_against_avg"):
                            feat[f"combined_{stat}_{window}g"] = (hv or 0) + (av or 0)
                    feat[f"home_game_total_avg_{window}g"] = (
                        (h_stats.get(f"pts_for_avg_{window}g", 0) or 0)
                        + (h_stats.get(f"pts_against_avg_{window}g", 0) or 0)
                    )
                    feat[f"away_game_total_avg_{window}g"] = (
                        (a_stats.get(f"pts_for_avg_{window}g", 0) or 0)
                        + (a_stats.get(f"pts_against_avg_{window}g", 0) or 0)
                    )
                feat["combined_season_pts_for"] = (
                    h_stats.get("season_point_diff_avg", 0.0) or 0.0
                ) + (a_stats.get("season_point_diff_avg", 0.0) or 0.0)
                feat["combined_rest"] = feat["home_rest"] + feat["away_rest"]
                for qb_col in qb_roll_cols:
                    hv = feat.get(f"home_{qb_col}", 0.0) or 0.0
                    av = feat.get(f"away_{qb_col}", 0.0) or 0.0
                    feat[f"combined_{qb_col}"] = hv + av
                feat["combined_injury_score"]     = feat["home_injury_score"]     + feat["away_injury_score"]
                feat["combined_qb_injury_impact"] = feat["home_qb_injury_impact"] + feat["away_qb_injury_impact"]

                aligned = align_features(feat, bundle)
                if aligned is not None:
                    prob_home = predict_proba(bundle, aligned)[0]
            else:
                print(f"  Missing stats: {home_api} ({home_abbr}) or {away_api} ({away_abbr})")
        else:
            print(f"  Unknown team mapping: {home_api} or {away_api}")

        pred = build_game_prediction(
            game, prob_home,
            best_line(game, "home"), best_line(game, "away"),
            "americanfootball_nfl", "NFL"
        )
        if pred.get("best_bet") and feat and prob_home is not None:
            pred["justification"] = _nfl_justification(
                home_api, away_api, feat, float(prob_home), pred["best_bet"],
                home_abbr=home_abbr, away_abbr=away_abbr,
                injury_scores=injury_scores, qb_stats=qb_stats,
            )
        # Surface injury data for dashboard
        pred["home_injury_score"]     = feat.get("home_injury_score", 0.0)
        pred["away_injury_score"]     = feat.get("away_injury_score", 0.0)
        pred["home_qb_injury_impact"] = feat.get("home_qb_injury_impact", 0.0)
        pred["away_qb_injury_impact"] = feat.get("away_qb_injury_impact", 0.0)

        # ── Over/Under prediction ─────────────────────────────────────────────
        if totals_bundle and feat:
            from models.trainer import predict_total, over_under_probs
            market_total = pred.get("total")
            totals_aligned = align_features(feat, totals_bundle)
            if totals_aligned is not None:
                predicted_total = float(predict_total(totals_bundle, totals_aligned)[0])
                pred["model_predicted_total"] = round(predicted_total, 1)
                if market_total and market_total.get("line"):
                    line = market_total["line"]
                    rmse = totals_bundle["rmse"]
                    p_over, p_under = over_under_probs(predicted_total, line, rmse)
                    over_odds  = market_total.get("over")
                    under_odds = market_total.get("under")

                    def _ou_no_vig(o_odds, u_odds):
                        if o_odds is None or u_odds is None:
                            return None, None
                        from utils.odds import remove_vig
                        return remove_vig(o_odds, u_odds)

                    mkt_over_nv, mkt_under_nv = _ou_no_vig(over_odds, under_odds)

                    ou_edge_over  = round(p_over  - mkt_over_nv,  4) if mkt_over_nv  else None
                    ou_edge_under = round(p_under - mkt_under_nv, 4) if mkt_under_nv else None

                    def _ou_ev(model_p, odds):
                        if odds is None:
                            return None
                        payout = abs(odds)/100 if odds < 0 else odds/100
                        return round(model_p * payout - (1 - model_p), 4)

                    pred["totals"] = {
                        "market_line":   line,
                        "predicted_total": round(predicted_total, 1),
                        "p_over":        round(p_over,  4),
                        "p_under":       round(p_under, 4),
                        "market_over_nv":  round(mkt_over_nv,  4) if mkt_over_nv  else None,
                        "market_under_nv": round(mkt_under_nv, 4) if mkt_under_nv else None,
                        "over_edge":     ou_edge_over,
                        "under_edge":    ou_edge_under,
                        "over_ev":       _ou_ev(p_over,  over_odds),
                        "under_ev":      _ou_ev(p_under, under_odds),
                        "best_ou_bet":   _best_ou_bet(
                            line, predicted_total, p_over, p_under,
                            over_odds, under_odds, ou_edge_over, ou_edge_under
                        ),
                    }
                else:
                    pred["totals"] = {"predicted_total": round(predicted_total, 1)}

        results.append(pred)
        tot_str = ""
        if pred.get("totals") and pred["totals"].get("market_line"):
            t = pred["totals"]
            tot_str = f" | O/U {t['market_line']} → model={t['predicted_total']} over={t['p_over']:.0%}"
        print(
            f"  {away_api} @ {home_api}: model={pred['model_home_prob']} "
            f"edge={pred['home_edge']}{tot_str}"
        )

    return results


def _nfl_justification(home: str, away: str, feat: dict,
                        model_home_prob: float, best_bet: dict,
                        home_abbr: str = None, away_abbr: str = None,
                        injury_scores: dict = None, qb_stats=None) -> str:
    """1-3 sentence justification for an NFL best bet, including QB and injury context."""
    if best_bet is None:
        return ""
    side      = best_bet["side"]
    edge_pct  = round(best_bet["edge"] * 100, 1)
    fav_team  = home if side == "home" else away
    dog_team  = away if side == "home" else home
    pfx       = "home" if side == "home" else "away"
    opp_pfx   = "away" if side == "home" else "home"
    fav_abbr  = home_abbr if side == "home" else away_abbr
    dog_abbr  = away_abbr if side == "home" else home_abbr

    parts = []

    # QB injury — mention first since it's highest-impact
    if injury_scores and fav_abbr and dog_abbr:
        fav_inj = injury_scores.get(fav_abbr, {})
        dog_inj = injury_scores.get(dog_abbr, {})
        dog_qb_impact = dog_inj.get("qb_injury_impact", 0.0)
        fav_qb_impact = fav_inj.get("qb_injury_impact", 0.0)
        if dog_qb_impact >= 0.5:
            parts.append(f"{dog_team}'s QB is listed as Out or Doubtful (impact score {dog_qb_impact:.2f}), significantly hurting their outlook")
        elif dog_qb_impact >= 0.2:
            parts.append(f"{dog_team}'s QB is questionable (injury impact {dog_qb_impact:.2f})")
        if fav_qb_impact >= 0.5:
            parts.append(f"Note: {fav_team}'s QB is also injured (impact {fav_qb_impact:.2f}) — edge may be reduced")

    # QB efficiency edge (EPA per attempt is most predictive)
    h_epa = feat.get("home_qb_epa_per_att_4w", 0.0) or 0.0
    a_epa = feat.get("away_qb_epa_per_att_4w", 0.0) or 0.0
    fav_epa = h_epa if side == "home" else a_epa
    dog_epa = a_epa if side == "home" else h_epa
    if abs(fav_epa - dog_epa) >= 0.05 and fav_epa != 0.0:
        parts.append(
            f"{fav_team}'s QB has a {fav_epa:+.3f} EPA/attempt advantage over {dog_team} ({dog_epa:+.3f}) over the last 4 weeks"
        )

    # Recent point differential
    d_fav = feat.get(f"{pfx}_point_diff_avg_8g")
    d_dog = feat.get(f"{opp_pfx}_point_diff_avg_8g")
    if d_fav is not None and str(d_fav) != "nan" and d_dog is not None and not parts:
        parts.append(
            f"{fav_team}'s avg point differential over last 8 games is {d_fav:+.1f} vs {d_dog:+.1f}"
        )

    # Rest/bye week edge
    rest = feat.get("rest_advantage", 0) or 0
    if side == "away":
        rest = -rest
    if abs(rest) >= 7:
        rested = fav_team if rest > 0 else dog_team
        parts.append(f"{rested} have a significant rest advantage ({abs(int(rest))} extra days — likely a bye week)")

    # Non-QB injury load on the opponent
    if injury_scores and dog_abbr:
        dog_non_qb = injury_scores.get(dog_abbr, {}).get("injury_score", 0.0)
        if dog_non_qb >= 1.0:
            parts.append(f"{dog_team} are dealing with significant non-QB injuries (score {dog_non_qb:.1f})")

    parts.append(
        f"Model prices {fav_team} at "
        f"{round(model_home_prob*100 if side=='home' else (1-model_home_prob)*100, 1)}% "
        f"— a +{edge_pct}% edge over the market"
    )
    return ". ".join(parts[:3]) + "."


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
        if best_bet:
            side_prob = p_home if best_bet["side"]=="home" else (p_draw if best_bet["side"]=="draw" else p_away)
            pred["justification"] = generate_justification(
                home_api, away_api, feat, side_prob,
                best_bet, "soccer_fifa_world_cup"
            )
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
    sports = sports or ["nba", "mlb", "nfl"]

    print(f"Running predictions for {target_date} — sports: {sports}")
    all_games = []

    if "nba" in sports:
        all_games += predict_nba(api_key, target_date)
    if "mlb" in sports:
        all_games += predict_mlb(api_key, target_date)
    if "nfl" in sports:
        all_games += predict_nfl(api_key, target_date)
    if "soccer" in sports:
        all_games += predict_soccer(api_key, target_date)

    # Cap at 3 best bets per day — keep the highest-EV plays, clear the rest
    MAX_BEST_BETS = 3
    flagged = [g for g in all_games if g.get("best_bet")]
    flagged.sort(key=lambda g: g["best_bet"]["ev"], reverse=True)
    keep_ids = {id(g) for g in flagged[:MAX_BEST_BETS]}
    for g in all_games:
        if g.get("best_bet") and id(g) not in keep_ids:
            g["best_bet"] = None

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date":         target_date,
        "games":        all_games,
    }
    Path(output_path).write_text(json.dumps(result, indent=2))

    strong   = sum(1 for g in all_games if g.get("best_bet") and g["best_bet"]["strength"] == "strong")
    moderate = sum(1 for g in all_games if g.get("best_bet") and g["best_bet"]["strength"] == "moderate")
    print(f"\nWrote {len(all_games)} predictions to {output_path}")
    print(f"Strong edges: {strong}  |  Moderate edges: {moderate} (capped at {MAX_BEST_BETS} total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("ODDS_API_KEY", ""))
    parser.add_argument("--date",    default=None,  help="YYYY-MM-DD (defaults to today)")
    parser.add_argument("--output",  default=None,  help="Output file (defaults to predictions_YYYY-MM-DD.json)")
    parser.add_argument("--sport",   default=None,  help="nba or mlb (defaults to both)")
    args = parser.parse_args()

    if not args.api_key:
        print("Warning: no ODDS_API_KEY — will use ESPN for game list, no odds data")

    sports = [args.sport] if args.sport else ["nba", "mlb", "nfl", "soccer"]
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
