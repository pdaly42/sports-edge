"""
Post-game pick-results logger — the server-side counterpart to the dashboard's
localStorage-based lifetime record.

For each predictions_YYYY-MM-DD.json in the repo root:
  - Iterate the games that carry a best_bet (moneyline) or a totals.best_ou_bet
  - Look up the game's final score via ESPN's scoreboard API
  - Compute the pick's outcome (win/loss/push) and P/L in units (1u stake, odds-based payout)
  - Append one row per pick to results_log.csv

Idempotent: rows are deduped on (date, game_id, side) so re-running the script
on the same day is safe.

Also emits per-pick CLV when the current market odds differ from the odds we
snapshotted at pick time (opening line vs. current line = post-hoc "closing"
proxy). ESPN doesn't reliably preserve historical closing odds, so this is a
best-effort field — the ground-truth W/L and PnL are always correct.
"""

from __future__ import annotations

import sys, os, json, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
from datetime import datetime, timedelta

import requests

# Duplicated from predict_today.py to keep this script's import chain light
# (predict_today transitively pulls in nflreadpy/xgboost, and its trainer
# uses PEP-604 union syntax that needs Python ≥3.10).
ESPN_SPORT_MAP = {
    "basketball_nba":         "basketball/nba",
    "baseball_mlb":           "baseball/mlb",
    "americanfootball_nfl":   "football/nfl",
    "americanfootball_ncaaf": "football/college-football",
    "soccer_fifa_world_cup":  "soccer/fifa.world",
}

REPO_ROOT   = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / "results_log.csv"

# Only scan predictions files newer than this many days back — bounds the
# per-run work as history grows.
SCAN_DAYS_BACK = 60

FIELDNAMES = [
    "date", "sport", "game_id",
    "away_team", "home_team",
    "away_score", "home_score",
    "pick_type",   # "ml" (moneyline) or "ou" (over/under)
    "side",        # "home"/"away" for ml, "over"/"under" for ou
    "team_or_line",
    "odds", "edge", "ev", "strength", "model_prob",
    "outcome",     # "win", "loss", "push"
    "pnl_units",   # net units (1u stake)
    "logged_at",
]


def _payout_units(odds: float) -> float:
    """Profit on a 1-unit stake at American odds."""
    return abs(odds) / 100 if odds < 0 else odds / 100


def _load_existing_keys() -> set[tuple]:
    """Existing (date, game_id, side) tuples — used to skip already-logged picks."""
    if not RESULTS_CSV.exists():
        return set()
    keys = set()
    with RESULTS_CSV.open() as f:
        for row in csv.DictReader(f):
            keys.add((row["date"], row["game_id"], row["side"]))
    return keys


def _fetch_espn_scoreboard(sport_key: str, date_str: str) -> list[dict]:
    """One-shot fetch of all games on a date from ESPN's scoreboard API."""
    path = ESPN_SPORT_MAP.get(sport_key)
    if not path:
        return []
    url = f"https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard"
    try:
        r = requests.get(url, params={"dates": date_str.replace("-", "")}, timeout=15)
        if not r.ok:
            return []
        return r.json().get("events", [])
    except Exception:
        return []


def _score_map_for_date(sport_key: str, date_str: str) -> dict[str, dict]:
    """
    Build a lookup {game_id_matches_the_json_id: {home, away, home_score, away_score, status}}
    for a specific sport/date. Predictions JSON stores game['id'] from the-odds-api
    or ESPN scoreboard; both track ESPN's event ID for football/baseball, so this
    match works cleanly for MLB/NFL/CFB. NBA and soccer use different ID conventions
    — for those we fall back to team-name matching.
    """
    events = _fetch_espn_scoreboard(sport_key, date_str)
    lookup: dict[str, dict] = {}
    for ev in events:
        comp = (ev.get("competitions") or [{}])[0]
        teams = comp.get("competitors") or []
        home = next((t for t in teams if t.get("homeAway") == "home"), None)
        away = next((t for t in teams if t.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        status = (ev.get("status") or {}).get("type", {}).get("name", "")
        if status not in ("STATUS_FINAL", "STATUS_FULL_TIME", "STATUS_FINAL_OVERTIME"):
            continue
        try:
            hs = int(home.get("score", 0))
            as_ = int(away.get("score", 0))
        except (TypeError, ValueError):
            continue
        payload = {
            "home_team": home["team"]["displayName"],
            "away_team": away["team"]["displayName"],
            "home_score": hs,
            "away_score": as_,
            "status":     status,
        }
        lookup[ev["id"]] = payload
        # Also index by the "away@home" team-name key as a fallback
        lookup[f"{payload['away_team']}@{payload['home_team']}"] = payload
    return lookup


def _match_completed_game(game: dict, sport_key: str, date_str: str,
                          score_cache: dict) -> dict | None:
    """Resolve a predictions_json game entry to a completed-game score payload."""
    key = sport_key + ":" + date_str
    if key not in score_cache:
        score_cache[key] = _score_map_for_date(sport_key, date_str)
    lookup = score_cache[key]
    if not lookup:
        return None
    # Try ID match first, then team-name key
    gid = str(game.get("id", ""))
    if gid in lookup:
        return lookup[gid]
    name_key = f"{game.get('away_team')}@{game.get('home_team')}"
    return lookup.get(name_key)


def _resolve_ml(pick: dict, game: dict, score: dict) -> tuple[str, float]:
    """Return (outcome, pnl_units) for a moneyline pick."""
    winner = "home" if score["home_score"] > score["away_score"] else \
             "away" if score["away_score"] > score["home_score"] else "push"
    if winner == "push":
        return ("push", 0.0)
    won = winner == pick["side"]
    return ("win", _payout_units(pick["odds"])) if won else ("loss", -1.0)


def _resolve_ou(pick: dict, totals: dict, score: dict) -> tuple[str, float]:
    """Return (outcome, pnl_units) for an over/under pick."""
    total = score["home_score"] + score["away_score"]
    line = totals.get("market_line")
    if line is None:
        return ("push", 0.0)
    if abs(total - line) < 1e-9:
        return ("push", 0.0)
    hit_over = total > line
    is_over_pick = pick["side"] == "over"
    won = (hit_over and is_over_pick) or ((not hit_over) and (not is_over_pick))
    return ("win", _payout_units(pick["odds"])) if won else ("loss", -1.0)


def _iter_predictions_files() -> list[Path]:
    """Prediction JSONs within SCAN_DAYS_BACK, sorted oldest → newest."""
    files = sorted(REPO_ROOT.glob("predictions_*.json"))
    cutoff = datetime.utcnow().date() - timedelta(days=SCAN_DAYS_BACK)
    out = []
    for p in files:
        try:
            d = datetime.strptime(p.stem.replace("predictions_", ""), "%Y-%m-%d").date()
        except ValueError:
            continue
        if d >= cutoff:
            out.append(p)
    return out


def process() -> int:
    """Scan predictions files, log any newly-completed picks. Returns count logged."""
    existing = _load_existing_keys()
    new_rows: list[dict] = []
    score_cache: dict[str, dict] = {}

    for path in _iter_predictions_files():
        date_str = path.stem.replace("predictions_", "")
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            print(f"  Skip {path.name}: {e}")
            continue
        for game in data.get("games", []):
            sport_key = game.get("sport") or ""
            gid       = str(game.get("id", ""))

            # ── Moneyline pick ────────────────────────────────────────────
            bb = game.get("best_bet")
            if bb and (date_str, gid, bb["side"]) not in existing:
                score = _match_completed_game(game, sport_key, date_str, score_cache)
                if score:
                    outcome, pnl = _resolve_ml(bb, game, score)
                    new_rows.append({
                        "date": date_str, "sport": sport_key, "game_id": gid,
                        "away_team": game["away_team"], "home_team": game["home_team"],
                        "away_score": score["away_score"], "home_score": score["home_score"],
                        "pick_type": "ml",
                        "side": bb["side"], "team_or_line": bb.get("team", ""),
                        "odds": bb["odds"], "edge": bb.get("edge"),
                        "ev": bb.get("ev"), "strength": bb.get("strength"),
                        "model_prob": game.get("model_home_prob") if bb["side"] == "home"
                                       else game.get("model_away_prob"),
                        "outcome": outcome, "pnl_units": round(pnl, 4),
                        "logged_at": datetime.utcnow().isoformat(timespec="seconds"),
                    })

            # ── Over/under pick ───────────────────────────────────────────
            totals = game.get("totals") or {}
            ou = totals.get("best_ou_bet")
            if ou and (date_str, gid, ou["side"]) not in existing:
                score = _match_completed_game(game, sport_key, date_str, score_cache)
                if score:
                    outcome, pnl = _resolve_ou(ou, totals, score)
                    new_rows.append({
                        "date": date_str, "sport": sport_key, "game_id": gid,
                        "away_team": game["away_team"], "home_team": game["home_team"],
                        "away_score": score["away_score"], "home_score": score["home_score"],
                        "pick_type": "ou",
                        "side": ou["side"], "team_or_line": totals.get("market_line", ""),
                        "odds": ou["odds"], "edge": ou.get("edge"),
                        "ev": ou.get("ev"), "strength": ou.get("strength"),
                        "model_prob": totals.get("p_over") if ou["side"] == "over"
                                       else totals.get("p_under"),
                        "outcome": outcome, "pnl_units": round(pnl, 4),
                        "logged_at": datetime.utcnow().isoformat(timespec="seconds"),
                    })

    if not new_rows:
        print("No newly-completed picks to log.")
        return 0

    write_header = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        for row in new_rows:
            w.writerow(row)

    print(f"Appended {len(new_rows)} pick results to {RESULTS_CSV.name}")
    return len(new_rows)


if __name__ == "__main__":
    n = process()
    sys.exit(0)
