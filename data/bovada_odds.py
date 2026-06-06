"""
Bovada public odds fetcher — no API key required.
Fetches from Bovada's internal website API, converts to the same
dict format the dashboard expects (matching the-odds-api.com structure).

Usage:
    from data.bovada_odds import fetch_bovada_odds
    games = fetch_bovada_odds("basketball_nba", "2026-06-05")
    # Returns list of game dicts with .bookmakers[] like the-odds-api.com format
"""

import requests
import warnings
from datetime import datetime, timezone
from typing import Optional

warnings.filterwarnings("ignore")

BASE = "https://www.bovada.lv/services/sports/event/coupon/events/A/description"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

SPORT_PATHS = {
    "basketball_nba":        "/basketball/nba",
    "baseball_mlb":          "/baseball/mlb",
    "americanfootball_nfl":  "/football/nfl",
    "soccer_fifa_world_cup": "/soccer/international",
}


def _parse_american(price_str: str):
    """Convert Bovada price string to integer, handling 'EVEN'."""
    if price_str is None:
        return None
    s = str(price_str).strip()
    if s.upper() == "EVEN":
        return 100
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def _parse_game(event: dict, sport_key: str, target_date: str) -> Optional[dict]:
    """
    Convert a Bovada event dict to the dashboard-compatible game dict.
    Returns None if the game isn't on target_date or has no main markets.
    """
    start_ms   = event.get("startTime", 0)
    start_dt   = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    game_date  = start_dt.date().isoformat()

    if game_date != target_date:
        return None

    desc = event.get("description", "")
    # Bovada format: "Away Team @ Home Team"
    if " @ " in desc:
        away_name, home_name = desc.split(" @ ", 1)
    else:
        return None

    away_name = away_name.strip()
    home_name = home_name.strip()

    # Collect markets from all display groups, skip quarter/half props
    skip = {"1H","1Q","2Q","3Q","4Q","2H","1P","2P","3P"}
    all_markets = []
    for dg in event.get("displayGroups", []):
        for m in dg.get("markets", []):
            desc_parts = m.get("description","").split(" - ")
            if any(s in m.get("description","") for s in skip):
                continue
            all_markets.append(m)

    h2h_market      = None
    spread_market   = None
    totals_market   = None

    for m in all_markets:
        d = m.get("description","")
        if d in ("Moneyline", "3-Way Moneyline") and h2h_market is None:
            h2h_market = m
        elif d in ("Point Spread", "Runline", "Goal Spread", "Run Line") and spread_market is None:
            spread_market = m
        elif d in ("Total", "Total Runs O/U", "Total Goals O/U") and totals_market is None:
            totals_market = m

    markets = []

    # h2h
    if h2h_market:
        outcomes = []
        for o in h2h_market.get("outcomes", []):
            odesc = o.get("description", "")
            price = _parse_american(o.get("price", {}).get("american"))
            if price is not None:
                # Normalise "Draw" — Bovada sometimes uses "TIE"
                name = "Draw" if odesc.upper() in ("DRAW","TIE") else odesc
                outcomes.append({"name": name, "price": price})
        if outcomes:
            markets.append({"key": "h2h", "outcomes": outcomes})

    # spreads
    if spread_market:
        outcomes = []
        for o in spread_market.get("outcomes", []):
            odesc = o.get("description","")
            price = _parse_american(o.get("price",{}).get("american"))
            # Spread point is often embedded in the outcome description e.g. "+5.5" or in handicap
            handicap = o.get("price", {}).get("handicap") or o.get("handicap")
            point = None
            if handicap is not None:
                try:
                    point = float(handicap)
                except (TypeError, ValueError):
                    pass
            if price is not None and odesc not in ("DRAW","TIE","Draw"):
                entry = {"name": odesc, "price": price}
                if point is not None:
                    entry["point"] = point
                outcomes.append(entry)
        if outcomes:
            markets.append({"key": "spreads", "outcomes": outcomes})

    # totals
    if totals_market:
        outcomes = []
        for o in totals_market.get("outcomes", []):
            odesc = o.get("description","").strip()
            price = _parse_american(o.get("price",{}).get("american"))
            point_str = o.get("price",{}).get("handicap") or o.get("handicap")
            point = None
            if point_str is not None:
                try:
                    point = float(point_str)
                except (TypeError, ValueError):
                    pass
            if price is not None and odesc.upper() in ("OVER","UNDER","O","U"):
                name = "Over" if odesc.upper() in ("OVER","O") else "Under"
                entry = {"name": name, "price": price}
                if point is not None:
                    entry["point"] = point
                outcomes.append(entry)
        if outcomes:
            markets.append({"key": "totals", "outcomes": outcomes})

    if not markets:
        return None

    return {
        "id":             f"bovada_{sport_key}_{away_name}_{home_name}".replace(" ","_"),
        "sport_key":      sport_key,
        "commence_time":  start_dt.isoformat(),
        "home_team":      home_name,
        "away_team":      away_name,
        "bookmakers": [{
            "key":     "bovada",
            "title":   "Bovada",
            "markets": markets,
        }],
    }


def fetch_bovada_odds(sport_key: str, target_date: str) -> list:
    """
    Fetch Bovada odds for a sport/date.
    Returns list of game dicts compatible with the-odds-api.com format.
    """
    path = SPORT_PATHS.get(sport_key)
    if not path:
        return []

    url = BASE + path
    try:
        r = requests.get(url, headers=HEADERS, params={"lang": "en"}, timeout=15)
        if r.status_code != 200:
            print(f"  Bovada {sport_key}: HTTP {r.status_code}")
            return []
        data = r.json()
    except Exception as e:
        print(f"  Bovada {sport_key}: {e}")
        return []

    events = [e for league in data for e in league.get("events", [])]
    games = []
    for ev in events:
        g = _parse_game(ev, sport_key, target_date)
        if g:
            games.append(g)

    print(f"  Bovada {sport_key}: {len(games)} games on {target_date}")
    return games
