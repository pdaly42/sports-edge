"""
One-off + safe-to-rerun backfill for results_log.csv.

Rebuilds coverage for picks that log_pick_results.py couldn't resolve, mostly
because of:

  1. Its 60-day scan cutoff (June-July MLB and World Cup picks aged out before
     server-side tracking was implemented).
  2. Home/away disagreement between the-odds-api and ESPN — e.g., the odds
     feed listed "Kansas Jayhawks @ Arizona State Sun Devils" while ESPN had
     the sides flipped. The prod resolver only tries one orientation.
  3. Minor team-name variants ("San José State" vs "San Jose State", trailing
     mascots, etc.).

Strategy for this script:

  - Scan ALL predictions_YYYY-MM-DD.json files, not just the last 60 days.
  - For each still-unresolved (date, game_id, side) triple, fetch ESPN's
     scoreboard for that (sport, date) and match games via
     ASCII-folded / prefix-based team-name matching instead of exact strings
     or the odds-api home/away orientation.
  - When a match is found, resolve the pick by *team identity* — did the team
     I picked win / cover / hit the over? — not by "did the home_or_away side
     I bet win?". That sidesteps the orientation-flip bug entirely.
  - MLB Statcast fallback: if ESPN has no match for an old MLB game (some old
     minor-league or spring-training exhibitions leak into predictions files),
     try the MLB StatsAPI, which is a more authoritative baseball source.

Safe to rerun: dedup on (date, game_id, pick_dedup_side) exactly like
log_pick_results.py, so already-logged picks are skipped.
"""

from __future__ import annotations

import sys
import os
import json
import csv
import re
import unicodedata
from pathlib import Path
from datetime import datetime

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Reuse constants and helpers from the prod logger where possible
from log_pick_results import (  # type: ignore
    ESPN_SPORT_MAP,
    FIELDNAMES,
    RESULTS_CSV,
    _payout_units,
    _load_existing_keys,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────
# Team name normalization
# ─────────────────────────────────────────────────────────────

def _fold(name: str) -> str:
    """Lowercase, strip diacritics, drop punctuation — for fuzzy team match."""
    if not name:
        return ""
    n = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower()
    n = re.sub(r"[^a-z0-9 ]+", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


# Common mascot / suffix tokens to drop when both sides pass a "core name"
# check but the ESPN full name has an extra word (e.g., "Kansas Jayhawks"
# vs "Kansas"). We progressively strip these to widen the match.
_MASCOT_SUFFIXES = {
    # Colleges — the long tail is huge; we only need the ones that actually
    # appear as differences between the odds feed and ESPN.
    "jayhawks", "sun devils", "wildcats", "tigers", "bulldogs", "hoosiers",
    "razorbacks", "gators", "seminoles", "hurricanes", "volunteers",
    "commodores", "gamecocks", "spartans", "wolverines", "buckeyes",
    "nittany lions", "fighting irish", "mountaineers", "cavaliers",
    "cardinals", "aggies", "longhorns", "cowboys", "cyclones", "sooners",
    "trojans", "bruins", "beavers", "ducks", "huskies", "cougars", "bears",
    "golden bears", "utes", "utah utes", "rams", "aztecs", "rebels",
    "hilltoppers", "roadrunners", "mustangs", "green wave", "owls",
    "chanticleers", "blue hens", "hokies", "yellow jackets", "eagles",
    "scarlet knights", "orange", "panthers", "mountaineers",
    # Pros — mostly a no-op since odds-api and ESPN both use full names,
    # but "Athletics" vs "Oakland Athletics" needs handling.
    "yankees", "red sox", "blue jays", "orioles", "rays",
    "royals", "twins", "guardians", "tigers", "white sox",
    "astros", "mariners", "rangers", "angels", "athletics",
    "mets", "phillies", "braves", "marlins", "nationals",
    "cubs", "cardinals", "brewers", "reds", "pirates",
    "dodgers", "giants", "padres", "rockies", "diamondbacks",
    # NFL
    "cowboys", "eagles", "giants", "commanders",
    "packers", "vikings", "bears", "lions",
    "buccaneers", "falcons", "panthers", "saints",
    "49ers", "seahawks", "rams", "cardinals",
    "bills", "dolphins", "patriots", "jets",
    "ravens", "bengals", "browns", "steelers",
    "texans", "colts", "jaguars", "titans",
    "broncos", "chiefs", "raiders", "chargers",
    # Soccer / FIFA — usually just the country/city name matches
}


def _strip_mascot(folded: str) -> str:
    """Try progressively-shorter forms of a folded name to help fuzzy match."""
    for suffix in sorted(_MASCOT_SUFFIXES, key=len, reverse=True):
        if folded.endswith(" " + suffix):
            return folded[: -(len(suffix) + 1)].strip()
    return folded


def _names_match(a: str, b: str) -> bool:
    """Two team names match if either equals the other after fold+mascot-strip,
    or if one is a prefix of the other on the folded form (handles cases
    where odds-api uses the short school name and ESPN uses the mascot form).
    """
    fa, fb = _fold(a), _fold(b)
    if not fa or not fb:
        return False
    if fa == fb:
        return True
    sa, sb = _strip_mascot(fa), _strip_mascot(fb)
    if sa and (sa == fb or sa == sb):
        return True
    if sb and (sb == fa or sb == sa):
        return True
    # Prefix match on the folded form as a last resort
    if fa.startswith(fb) or fb.startswith(fa):
        # Only accept if the prefix is long enough to be distinctive
        return min(len(fa), len(fb)) >= 4
    return False


# ─────────────────────────────────────────────────────────────
# Score sources
# ─────────────────────────────────────────────────────────────

def _fetch_espn_events(sport_key: str, date_str: str) -> list[dict]:
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


def _fetch_mlb_statsapi(date_str: str) -> list[dict]:
    """MLB's official StatsAPI — reliable for historical dates that ESPN
    may have purged. Returns a normalized list of finished games."""
    url = "https://statsapi.mlb.com/api/v1/schedule"
    try:
        r = requests.get(url, params={
            "sportId": 1,
            "date": date_str,
            "hydrate": "linescore,team",
        }, timeout=15)
        if not r.ok:
            return []
        js = r.json()
    except Exception:
        return []
    games: list[dict] = []
    for d in js.get("dates", []):
        for g in d.get("games", []):
            status = g.get("status", {}).get("abstractGameState", "")
            if status != "Final":
                continue
            teams = g.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            try:
                hs = int(home.get("score"))
                as_ = int(away.get("score"))
            except (TypeError, ValueError):
                continue
            games.append({
                "home_team": home.get("team", {}).get("name", ""),
                "away_team": away.get("team", {}).get("name", ""),
                "home_score": hs,
                "away_score": as_,
            })
    return games


def _events_normalize(sport_key: str, events: list[dict]) -> list[dict]:
    """Reduce ESPN raw events to the fields we need."""
    out = []
    for ev in events:
        comp = (ev.get("competitions") or [{}])[0]
        teams = comp.get("competitors") or []
        home = next((t for t in teams if t.get("homeAway") == "home"), None)
        away = next((t for t in teams if t.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        status = (ev.get("status") or {}).get("type", {}).get("name", "")
        # STATUS_FINAL_PEN / STATUS_FINAL_AET are World Cup knockout finishes.
        # For moneyline, the ESPN "score" field reflects goals at the end of
        # the shown status (i.e., includes ET goals for AET, excludes pens for
        # PEN) — a PEN result at 1-1 correctly resolves as a push under our
        # tie-goes-to-push moneyline convention.
        if status not in (
            "STATUS_FINAL",
            "STATUS_FULL_TIME",
            "STATUS_FINAL_OVERTIME",
            "STATUS_FINAL_PEN",
            "STATUS_FINAL_AET",
        ):
            continue
        try:
            hs = int(home.get("score", 0))
            as_ = int(away.get("score", 0))
        except (TypeError, ValueError):
            continue
        out.append({
            "home_team": home["team"]["displayName"],
            "away_team": away["team"]["displayName"],
            "home_score": hs,
            "away_score": as_,
        })
    return out


# Cache: (sport_key, date_str) → list of normalized finished games
_score_cache: dict[tuple, list[dict]] = {}


def _finished_games(sport_key: str, date_str: str) -> list[dict]:
    key = (sport_key, date_str)
    if key not in _score_cache:
        games = _events_normalize(sport_key, _fetch_espn_events(sport_key, date_str))
        # MLB fallback
        if sport_key == "baseball_mlb" and not games:
            games = _fetch_mlb_statsapi(date_str)
        _score_cache[key] = games
    return _score_cache[key]


def _find_game(sport_key: str, date_str: str, away_team: str, home_team: str) -> dict | None:
    """Fuzzy-match a predictions_json game entry against finished-games list.
    Tries: exact-orientation, swapped-orientation, then per-team fuzzy."""
    games = _finished_games(sport_key, date_str)
    if not games:
        return None
    for g in games:
        if _names_match(g["away_team"], away_team) and _names_match(g["home_team"], home_team):
            return g
    # Swapped orientation — odds-api may disagree with ESPN on home vs away.
    # If we match with swap, flip the game payload so downstream side logic
    # is consistent with the odds-api's home/away convention.
    for g in games:
        if _names_match(g["away_team"], home_team) and _names_match(g["home_team"], away_team):
            return {
                "home_team": g["away_team"],
                "away_team": g["home_team"],
                "home_score": g["away_score"],
                "away_score": g["home_score"],
            }
    return None


# ─────────────────────────────────────────────────────────────
# Pick resolvers (mirror log_pick_results.py but score-source agnostic)
# ─────────────────────────────────────────────────────────────

def _resolve_ml(pick: dict, score: dict) -> tuple[str, float]:
    winner = "home" if score["home_score"] > score["away_score"] else \
             "away" if score["away_score"] > score["home_score"] else "push"
    if winner == "push":
        return ("push", 0.0)
    won = winner == pick["side"]
    return ("win", _payout_units(pick["odds"])) if won else ("loss", -1.0)


def _resolve_ou(pick: dict, totals: dict, score: dict) -> tuple[str, float]:
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


def _resolve_ats(pick: dict, sa: dict, score: dict) -> tuple[str, float]:
    home_line = sa.get("line")
    if home_line is None:
        return ("push", 0.0)
    margin = (score["home_score"] - score["away_score"]) + float(home_line)
    if abs(margin) < 1e-9:
        return ("push", 0.0)
    home_covers = margin > 0
    is_home_pick = pick["side"] == "home"
    won = (home_covers and is_home_pick) or ((not home_covers) and (not is_home_pick))
    return ("win", _payout_units(pick["odds"])) if won else ("loss", -1.0)


# ─────────────────────────────────────────────────────────────
# Main pass
# ─────────────────────────────────────────────────────────────

def process(*, dry_run: bool = False, verbose: bool = False) -> tuple[int, list[dict]]:
    existing = _load_existing_keys()
    new_rows: list[dict] = []
    misses: list[dict] = []
    files = sorted(REPO_ROOT.glob("predictions_*.json"))
    print(f"Scanning {len(files)} predictions files (no date cutoff)…")

    for path in files:
        date_str = path.stem.replace("predictions_", "")
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            print(f"  Skip {path.name}: {e}")
            continue

        for game in data.get("games", []):
            sport_key = game.get("sport") or ""
            gid = str(game.get("id", ""))
            away = game.get("away_team", "")
            home = game.get("home_team", "")

            # Which picks in this game still need resolving?
            todo = []
            bb = game.get("best_bet")
            if bb and (date_str, gid, bb["side"]) not in existing:
                todo.append(("ml", bb))
            ou = (game.get("totals") or {}).get("best_ou_bet")
            if ou and (date_str, gid, ou["side"]) not in existing:
                todo.append(("ou", ou))
            ats = (game.get("spread_analysis") or {}).get("best_ats_bet")
            if ats and (date_str, gid, f"ats-{ats['side']}") not in existing:
                todo.append(("ats", ats))

            if not todo:
                continue

            score = _find_game(sport_key, date_str, away, home)
            if not score:
                for kind, pick in todo:
                    misses.append({
                        "date": date_str, "sport": sport_key,
                        "away": away, "home": home, "kind": kind,
                    })
                if verbose:
                    print(f"  MISS {date_str} {sport_key}: {away} @ {home}")
                continue

            for kind, pick in todo:
                if kind == "ml":
                    outcome, pnl = _resolve_ml(pick, score)
                    row_side = pick["side"]
                    team_or_line = pick.get("team", "")
                    model_prob = game.get("model_home_prob") if pick["side"] == "home" \
                                  else game.get("model_away_prob")
                elif kind == "ou":
                    totals = game.get("totals") or {}
                    outcome, pnl = _resolve_ou(pick, totals, score)
                    row_side = pick["side"]
                    team_or_line = totals.get("market_line", "")
                    model_prob = totals.get("p_over") if pick["side"] == "over" \
                                  else totals.get("p_under")
                else:  # ats
                    sa = game.get("spread_analysis") or {}
                    outcome, pnl = _resolve_ats(pick, sa, score)
                    row_side = f"ats-{pick['side']}"
                    team_or_line = f"{pick.get('team','?')} {pick.get('line','?'):+}" \
                        if isinstance(pick.get("line"), (int, float)) else f"{pick.get('team','?')} ?"
                    model_prob = sa.get("model_p_home_covers") if pick["side"] == "home" \
                                  else sa.get("model_p_away_covers")

                new_rows.append({
                    "date": date_str, "sport": sport_key, "game_id": gid,
                    "away_team": away, "home_team": home,
                    "away_score": score["away_score"], "home_score": score["home_score"],
                    "pick_type": kind,
                    "side": row_side,
                    "team_or_line": team_or_line,
                    "odds": pick.get("odds"),
                    "edge": pick.get("edge"),
                    "ev": pick.get("ev"),
                    "strength": pick.get("strength"),
                    "model_prob": model_prob,
                    "outcome": outcome, "pnl_units": round(pnl, 4),
                    "logged_at": datetime.utcnow().isoformat(timespec="seconds"),
                })

    print(f"\nResolved {len(new_rows)} previously-unlogged picks.")
    print(f"Still-missing (no ESPN/StatsAPI match): {len(misses)}")

    # Aggregate misses by sport for the ops report
    if misses:
        from collections import Counter
        by_sport = Counter(m["sport"] for m in misses)
        print(f"  Misses by sport: {dict(by_sport)}")
        if verbose:
            for m in misses[:20]:
                print(f"    {m['date']} {m['sport']}: {m['away']} @ {m['home']} ({m['kind']})")

    if new_rows and not dry_run:
        write_header = not RESULTS_CSV.exists()
        with RESULTS_CSV.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                w.writeheader()
            for r in new_rows:
                w.writerow(r)
        print(f"Appended {len(new_rows)} rows to {RESULTS_CSV.name}.")
    elif dry_run:
        print("(dry-run: not writing)")

    return len(new_rows), misses


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Analyze but don't write to CSV")
    ap.add_argument("--verbose", action="store_true", help="Print each miss and near-match")
    args = ap.parse_args()
    process(dry_run=args.dry_run, verbose=args.verbose)
