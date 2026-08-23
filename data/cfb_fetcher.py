"""
College football data via CollegeFootballData.com REST API.

Scope filter: we only train and predict on games where the market is
reasonably sharp — Power 4 conference games, and any game where at least
one team is AP top-25. This concentrates the model on the ~40-60 games
per Saturday that matter for betting, and avoids the FBS-vs-FCS blowout
noise that dominates raw CFBD data.

Requires CFBD_API_KEY env var (free key at collegefootballdata.com/key).
"""

import pandas as pd
import numpy as np
import sys, os
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import requests
from pathlib import Path

from config.settings import RAW_DIR, CFBD_API_KEY
from utils.features import rolling_avg

CFB_RAW = RAW_DIR / "cfb"
CFB_RAW.mkdir(parents=True, exist_ok=True)

CFBD_BASE = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}"} if CFBD_API_KEY else {}

# Power 4 conferences per CFBD's `conference` field.
# Notre Dame is Independent but plays a P4-caliber schedule year-round.
P4_CONFERENCES = {"SEC", "Big Ten", "ACC", "Big 12"}
P4_INDEPENDENTS = {"Notre Dame"}

DEFAULT_SEASONS = list(range(2015, datetime.now().year + 1))


def _get(path: str, params: dict = None) -> list:
    """Thin GET wrapper — returns [] on any failure, logs the reason."""
    if not CFBD_API_KEY:
        print(f"  CFBD: no API key set (CFBD_API_KEY) — cannot fetch {path}")
        return []
    try:
        r = requests.get(f"{CFBD_BASE}{path}", headers=HEADERS, params=params or {}, timeout=30)
        if not r.ok:
            print(f"  CFBD {path}: HTTP {r.status_code} — {r.text[:120]}")
            return []
        return r.json()
    except Exception as e:
        print(f"  CFBD {path}: network error — {e}")
        return []


def fetch_season_games(season: int) -> pd.DataFrame:
    """Fetch all regular-season FBS games for one season. Cached per season."""
    cache_path = CFB_RAW / f"games_{season}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["start_date"])

    print(f"  Fetching CFBD games for {season}...")
    raw = _get("/games", {"year": season, "seasonType": "regular", "division": "fbs"})
    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    keep = [
        "id", "season", "week", "season_type", "start_date",
        "home_team", "away_team", "home_conference", "away_conference",
        "home_points", "away_points", "neutral_site", "conference_game",
        "venue",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["start_date"] = pd.to_datetime(df["start_date"], utc=True).dt.tz_convert(None)

    df.to_csv(cache_path, index=False)
    print(f"  Saved {len(df)} games for {season}")
    return df


def fetch_all_games(seasons: list = None) -> pd.DataFrame:
    """Concatenate season fetches."""
    seasons = seasons or DEFAULT_SEASONS
    frames = [fetch_season_games(s) for s in seasons]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_rankings(season: int) -> pd.DataFrame:
    """
    AP + CFP weekly rankings. Returns long-form: (season, week, team, poll, rank).
    Cached per season.
    """
    cache_path = CFB_RAW / f"rankings_{season}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    raw = _get("/rankings", {"year": season, "seasonType": "regular"})
    rows = []
    for wk in raw:
        for poll in wk.get("polls", []):
            if poll["poll"] not in ("AP Top 25", "Playoff Committee Rankings"):
                continue
            for r in poll.get("ranks", []):
                rows.append({
                    "season": wk["season"], "week": wk["week"],
                    "poll": poll["poll"], "team": r["school"], "rank": r["rank"],
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(cache_path, index=False)
    return df


def _is_qualifying_game(row: pd.Series, ranked_teams_by_week: dict) -> bool:
    """
    A game qualifies for the model if:
      - both teams are Power 4 (or Notre Dame), OR
      - at least one team is AP top-25 in the week of the game.
    """
    home_conf = row.get("home_conference") or ""
    away_conf = row.get("away_conference") or ""
    home_team = row["home_team"]
    away_team = row["away_team"]

    home_p4 = home_conf in P4_CONFERENCES or home_team in P4_INDEPENDENTS
    away_p4 = away_conf in P4_CONFERENCES or away_team in P4_INDEPENDENTS
    if home_p4 and away_p4:
        return True

    ranked = ranked_teams_by_week.get((row["season"], row["week"]), set())
    if home_team in ranked or away_team in ranked:
        return True
    return False


def build_team_game_log(seasons: list = None) -> pd.DataFrame:
    """
    Per-team-per-game log with rolling engineered features.
    Includes ALL FBS games (not just qualifying) so rolling stats reflect
    the team's real schedule, not just the games we'd bet on.
    """
    seasons = seasons or DEFAULT_SEASONS
    raw = fetch_all_games(seasons)
    if raw.empty:
        return pd.DataFrame()

    # Only completed games
    raw = raw[raw["home_points"].notna() & raw["away_points"].notna()].copy()
    raw["home_points"] = raw["home_points"].astype(int)
    raw["away_points"] = raw["away_points"].astype(int)
    raw = raw.rename(columns={"start_date": "date"})

    home = raw.copy()
    home["team"]          = home["home_team"]
    home["opponent"]      = home["away_team"]
    home["points_for"]    = home["home_points"]
    home["points_against"] = home["away_points"]
    home["is_home"]       = 1

    away = raw.copy()
    away["team"]          = away["away_team"]
    away["opponent"]      = away["home_team"]
    away["points_for"]    = away["away_points"]
    away["points_against"] = away["home_points"]
    away["is_home"]       = 0

    keep = ["date", "season", "week", "id", "team", "opponent", "is_home",
            "points_for", "points_against", "conference_game", "neutral_site"]
    log = pd.concat([home[keep], away[keep]], ignore_index=True)
    log["win"]        = (log["points_for"] > log["points_against"]).astype(int)
    log["point_diff"] = log["points_for"] - log["points_against"]
    log = log.sort_values(["team", "date"]).reset_index(drop=True)

    # Days rest per team (game-to-game gap; NaN on the first game of a season)
    log["days_rest"] = log.groupby("team")["date"].transform(
        lambda x: pd.to_datetime(x).diff().dt.days
    )
    log["days_rest"] = log["days_rest"].clip(upper=21).fillna(7)

    # Rolling stats — CFB seasons are short (12-15 games) so use 3g/6g windows
    for window in [3, 6]:
        log[f"win_pct_{window}g"]         = rolling_avg(log, "win",            window)
        log[f"pts_for_avg_{window}g"]     = rolling_avg(log, "points_for",     window)
        log[f"pts_against_avg_{window}g"] = rolling_avg(log, "points_against", window)
        log[f"point_diff_avg_{window}g"]  = rolling_avg(log, "point_diff",     window)

    # Season-to-date (shift(1) prevents leakage)
    log["season_win_pct"] = (
        log.groupby(["team", "season"])["win"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    log["season_point_diff_avg"] = (
        log.groupby(["team", "season"])["point_diff"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    return log


def build_matchup_features(seasons: list = None) -> pd.DataFrame:
    """
    Per-game matchup DataFrame for training the CFB model.
    Target: home_win (1 = home team won).
    Filters to qualifying games (P4 + top-25) only.
    Merges in SP+ ratings and team EPA where available.
    """
    seasons = seasons or DEFAULT_SEASONS
    log = build_team_game_log(seasons)
    if log.empty:
        return pd.DataFrame()

    home_log = log[log["is_home"] == 1].copy()
    away_log = log[log["is_home"] == 0].copy()

    roll_cols = [c for c in log.columns if any(
        c.startswith(p) for p in
        ["win_pct_", "pts_for_avg_", "pts_against_avg_", "point_diff_avg_",
         "season_win_pct", "season_point_diff_avg"]
    )]

    home_feats = home_log[["date", "season", "week", "id", "team", "opponent",
                           "win", "days_rest", "points_for", "points_against",
                           "neutral_site"] + roll_cols].copy()
    home_feats.columns = (
        ["date", "season", "week", "id", "home_team", "away_team",
         "home_win", "home_days_rest", "home_points", "away_points",
         "neutral_site"]
        + [f"home_{c}" for c in roll_cols]
    )

    away_feats = away_log[["date", "team", "days_rest"] + roll_cols].copy()
    away_feats.columns = (
        ["date", "away_team", "away_days_rest"]
        + [f"away_{c}" for c in roll_cols]
    )

    matchups = home_feats.merge(away_feats, on=["date", "away_team"])
    matchups["total_points"] = matchups["home_points"] + matchups["away_points"]

    # Differential features
    for window in [3, 6]:
        matchups[f"win_pct_diff_{window}g"] = (
            matchups[f"home_win_pct_{window}g"] - matchups[f"away_win_pct_{window}g"]
        )
        matchups[f"point_diff_diff_{window}g"] = (
            matchups[f"home_point_diff_avg_{window}g"] - matchups[f"away_point_diff_avg_{window}g"]
        )
    matchups["rest_advantage"] = matchups["home_days_rest"] - matchups["away_days_rest"]
    matchups["season_win_pct_diff"] = (
        matchups["home_season_win_pct"] - matchups["away_season_win_pct"]
    )
    matchups["season_point_diff_avg_diff"] = (
        matchups["home_season_point_diff_avg"] - matchups["away_season_point_diff_avg"]
    )

    # Totals-focused combined features
    for window in [3, 6]:
        matchups[f"combined_pts_for_{window}g"] = (
            matchups[f"home_pts_for_avg_{window}g"] + matchups[f"away_pts_for_avg_{window}g"]
        )
        matchups[f"combined_pts_against_{window}g"] = (
            matchups[f"home_pts_against_avg_{window}g"] + matchups[f"away_pts_against_avg_{window}g"]
        )

    # ── Filter to qualifying games (P4 or has ranked team) ────────────────────
    # We need the conferences from raw to know P4 status; re-merge those in.
    raw = fetch_all_games(seasons)
    if not raw.empty:
        conf = raw[["id", "home_conference", "away_conference"]].drop_duplicates("id")
        matchups = matchups.merge(conf, on="id", how="left")

    # Build ranked lookup
    ranked_by_week = {}
    for s in seasons:
        rk = fetch_rankings(s)
        if rk.empty:
            continue
        ap = rk[rk["poll"] == "AP Top 25"]
        for (season, week), group in ap.groupby(["season", "week"]):
            ranked_by_week.setdefault((season, week), set()).update(group["team"].tolist())

    mask = matchups.apply(lambda r: _is_qualifying_game(r, ranked_by_week), axis=1)
    matchups = matchups[mask].copy()
    print(f"  CFB qualifying-game filter: {mask.sum()} / {len(mask)} games kept")

    # ── SP+ ratings merge ──────────────────────────────────────────────────────
    try:
        from data.cfb_ratings import get_sp_features_for_matchups
        sp_df = get_sp_features_for_matchups(seasons)
        if not sp_df.empty:
            sp_cols = [c for c in sp_df.columns if c not in ("season", "team")]
            home_sp = sp_df.rename(columns={"team": "home_team"})
            home_sp = home_sp.rename(columns={c: f"home_{c}" for c in sp_cols})
            matchups = matchups.merge(home_sp, on=["season", "home_team"], how="left")
            away_sp = sp_df.rename(columns={"team": "away_team"})
            away_sp = away_sp.rename(columns={c: f"away_{c}" for c in sp_cols})
            matchups = matchups.merge(away_sp, on=["season", "away_team"], how="left")
            for col in sp_cols:
                matchups[f"sp_{col}_diff"] = matchups[f"home_{col}"] - matchups[f"away_{col}"]
            sp_feature_cols = (
                [f"home_{c}" for c in sp_cols]
                + [f"away_{c}" for c in sp_cols]
                + [f"sp_{c}_diff" for c in sp_cols]
            )
            matchups[sp_feature_cols] = matchups[sp_feature_cols].fillna(0.0)
            print(f"  SP+ features merged: {len(sp_cols)} cols per team")
    except Exception as e:
        print(f"  Warning: SP+ features skipped — {e}")

    # ── Team EPA merge (rolling 3w/6w, offense/defense) ───────────────────────
    try:
        from data.cfb_epa import get_team_epa_for_matchups
        epa_df = get_team_epa_for_matchups(seasons)
        if not epa_df.empty:
            epa_cols = [c for c in epa_df.columns if c not in ("season", "week", "team")]
            home_epa = epa_df.rename(columns={"team": "home_team"})
            home_epa = home_epa.rename(columns={c: f"home_{c}" for c in epa_cols})
            matchups = matchups.merge(home_epa, on=["season", "week", "home_team"], how="left")
            away_epa = epa_df.rename(columns={"team": "away_team"})
            away_epa = away_epa.rename(columns={c: f"away_{c}" for c in epa_cols})
            matchups = matchups.merge(away_epa, on=["season", "week", "away_team"], how="left")
            for col in epa_cols:
                matchups[f"{col}_diff"] = matchups[f"home_{col}"] - matchups[f"away_{col}"]
                matchups[f"combined_{col}"] = matchups[f"home_{col}"] + matchups[f"away_{col}"]
            epa_feature_cols = (
                [f"home_{c}" for c in epa_cols]
                + [f"away_{c}" for c in epa_cols]
                + [f"{c}_diff" for c in epa_cols]
                + [f"combined_{c}" for c in epa_cols]
            )
            matchups[epa_feature_cols] = matchups[epa_feature_cols].fillna(0.0)
            print(f"  CFB team EPA features merged: {len(epa_cols)} rolling cols per team")
    except Exception as e:
        print(f"  Warning: CFB team EPA features skipped — {e}")

    return matchups


def get_current_team_stats(seasons: list = None) -> pd.DataFrame:
    """Most recent rolling stats per team — for live predictions."""
    if seasons is None:
        current_year = datetime.now().year
        seasons = list(range(current_year - 2, current_year + 1))
    # Bust current-season cache so live picks reflect latest results
    for s in seasons:
        if s == datetime.now().year:
            p = CFB_RAW / f"games_{s}.csv"
            if p.exists():
                p.unlink()
    log = build_team_game_log(seasons)
    if log.empty:
        return pd.DataFrame()
    return log.sort_values("date").groupby("team").last().reset_index()


def get_current_ranked_teams(season: int = None, week: int = None) -> set:
    """AP top-25 for the given (season, week), or the latest available."""
    season = season or datetime.now().year
    # Bust cache so we get the freshest rankings
    p = CFB_RAW / f"rankings_{season}.csv"
    if p.exists():
        p.unlink()
    rk = fetch_rankings(season)
    if rk.empty:
        return set()
    ap = rk[rk["poll"] == "AP Top 25"]
    if week:
        ap = ap[ap["week"] == week]
    else:
        # Latest week available
        latest = ap["week"].max()
        ap = ap[ap["week"] == latest]
    return set(ap["team"].tolist())


if __name__ == "__main__":
    df = build_matchup_features(list(range(2020, datetime.now().year + 1)))
    print(f"Built {len(df)} CFB qualifying-game matchups")
    if not df.empty:
        print(df.head())
