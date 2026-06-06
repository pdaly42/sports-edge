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

def _needs_training(model_path: Path) -> bool:
    if not model_path.exists():
        return True
    age_days = (time.time() - model_path.stat().st_mtime) / 86400
    if age_days > RETRAIN_DAYS:
        print(f"  Model is {age_days:.1f} days old — retraining to include recent games")
        return True
    return False


def train_nba():
    model_path = Path('models/nba_xgb_model.pkl')
    if not _needs_training(model_path):
        print("NBA model is current, skipping")
        return
    print("Training NBA model (2019-2025 + current season)...")
    from data.nba_fetcher import build_matchup_features
    from models.trainer import train
    current_year = datetime.utcnow().year
    df = build_matchup_features(list(range(2019, current_year + 1)))
    df = df.dropna(subset=['home_win']).reset_index(drop=True)
    # Hold out current season from final CV but include all completed games for fitting
    train_df = df.copy()
    train(train_df, sport='nba', model_type='xgb')
    print("NBA model trained.")


def train_mlb():
    model_path = Path('models/mlb_xgb_model.pkl')
    if not _needs_training(model_path):
        print("MLB model is current, skipping")
        return
    print("Training MLB model (2019-2025 + current season)...")
    from data.mlb_fetcher import build_matchup_features
    from models.trainer import train
    current_year = datetime.utcnow().year
    df = build_matchup_features(list(range(2019, current_year + 1)))
    df = df.dropna(subset=['home_win']).reset_index(drop=True)
    train(df, sport='mlb', model_type='xgb')
    print("MLB model trained.")


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
    train_soccer()
