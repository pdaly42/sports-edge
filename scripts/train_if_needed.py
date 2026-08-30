"""
Train NBA and MLB models if their .pkl files don't exist or are older than RETRAIN_DAYS.
Called by the GitHub Actions workflow before running predictions.
"""
import sys, os, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime

# Retrain if model is older than this many days (picks up current-season results)
RETRAIN_DAYS = 7

# How many seasons of history to include in each model's training set.
# Rationale: roster/coaching turnover makes older data less representative.
# The recency-weighting inside trainer.train() further biases the model
# toward the current season within this window.
TRAIN_SEASONS_BACK = 2   # last 2 completed seasons + current = 3 total


def _needs_training(model_path: Path) -> bool:
    if not model_path.exists():
        return True
    age_days = (time.time() - model_path.stat().st_mtime) / 86400
    if age_days > RETRAIN_DAYS:
        print(f"  Model is {age_days:.1f} days old — retraining to include recent games")
        return True
    return False


def _recent_seasons(current_year: int, n_back: int = TRAIN_SEASONS_BACK) -> list[int]:
    """Rolling training window: current year + n_back completed seasons."""
    return list(range(current_year - n_back, current_year + 1))


def train_nba():
    model_path = Path('models/nba_xgb_model.pkl')
    if not _needs_training(model_path):
        print("NBA model is current, skipping")
        return
    from data.nba_fetcher import build_matchup_features
    from models.trainer import train
    current_year = datetime.utcnow().year
    seasons = _recent_seasons(current_year)
    print(f"Training NBA model (seasons {min(seasons)}-{max(seasons)}, "
          f"recency-weighted)...")
    df = build_matchup_features(seasons)
    df = df.dropna(subset=['home_win']).reset_index(drop=True)
    train(df, sport='nba', model_type='xgb', apply_recency_weights=True)
    print("NBA model trained.")


def train_mlb():
    model_path = Path('models/mlb_xgb_model.pkl')
    if not _needs_training(model_path):
        print("MLB model is current, skipping")
        return
    from data.mlb_fetcher import build_matchup_features
    from models.trainer import train
    current_year = datetime.utcnow().year
    seasons = _recent_seasons(current_year)
    print(f"Training MLB model (seasons {min(seasons)}-{max(seasons)}, "
          f"recency-weighted)...")
    df = build_matchup_features(seasons)
    df = df.dropna(subset=['home_win']).reset_index(drop=True)
    train(df, sport='mlb', model_type='xgb', apply_recency_weights=True)
    print("MLB model trained.")


def _build_nfl_matchups(current_year: int) -> object:
    from data.nfl_fetcher import build_matchup_features
    return build_matchup_features(_recent_seasons(current_year))


def train_nfl():
    model_path = Path('models/nfl_xgb_model.pkl')
    if not _needs_training(model_path):
        print("NFL winner model is current, skipping")
        return
    from models.trainer import train
    current_year = datetime.utcnow().year
    seasons = _recent_seasons(current_year)
    print(f"Training NFL winner model (seasons {min(seasons)}-{max(seasons)}, "
          f"recency-weighted)...")
    df = _build_nfl_matchups(current_year)
    df = df.dropna(subset=['home_win']).reset_index(drop=True)
    train(df, sport='nfl', model_type='xgb', apply_recency_weights=True)
    print("NFL winner model trained.")


def train_nfl_totals():
    model_path = Path('models/nfl_totals_model.pkl')
    if not _needs_training(model_path):
        print("NFL totals model is current, skipping")
        return
    from models.trainer import train_regression
    current_year = datetime.utcnow().year
    seasons = _recent_seasons(current_year)
    print(f"Training NFL totals model (seasons {min(seasons)}-{max(seasons)}, "
          f"recency-weighted)...")
    df = _build_nfl_matchups(current_year)
    df = df.dropna(subset=['total_points']).reset_index(drop=True)
    train_regression(df, target='total_points', sport='nfl', model_name='totals',
                     apply_recency_weights=True)
    print("NFL totals model trained.")


def train_cfb():
    model_path = Path('models/cfb_xgb_model.pkl')
    if not _needs_training(model_path):
        print("CFB model is current, skipping")
        return
    from data.cfb_fetcher import build_matchup_features
    from models.trainer import train
    current_year = datetime.utcnow().year
    seasons = _recent_seasons(current_year)
    print(f"Training CFB model (seasons {min(seasons)}-{max(seasons)}, "
          f"P4 + top-25 games, recency-weighted)...")
    df = build_matchup_features(seasons)
    if df is None or df.empty:
        print("  CFB matchup build returned no rows — is CFBD_API_KEY set? Skipping.")
        return
    df = df.dropna(subset=['home_win']).reset_index(drop=True)
    train(df, sport='cfb', model_type='xgb', apply_recency_weights=True)
    print("CFB model trained.")


def train_soccer():
    model_path = Path('models/soccer_wc_model.pkl')
    if not _needs_training(model_path):
        print("Soccer model is current, skipping")
        return
    print("Training Soccer/World Cup model...")
    from data.soccer_fetcher import build_matchup_features
    from models.soccer_trainer import train_soccer as _train
    df = build_matchup_features()
    _train(df, sport="soccer_wc")
    print("Soccer model trained.")


if __name__ == "__main__":
    train_nba()
    train_mlb()
    train_nfl()
    train_nfl_totals()
    train_cfb()
    train_soccer()
