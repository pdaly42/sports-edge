"""
Train NBA and MLB models if their .pkl files don't already exist.
Called by the GitHub Actions workflow before running predictions.
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings("ignore")

from pathlib import Path

def train_nba():
    model_path = Path('models/nba_xgb_model.pkl')
    if model_path.exists():
        print("NBA model already exists, skipping")
        return
    print("Training NBA model...")
    from data.nba_fetcher import build_matchup_features
    from models.trainer import train
    df = build_matchup_features(list(range(2019, 2025)))
    df = df.dropna(subset=['home_win']).reset_index(drop=True)
    train_df = df[df['season'] < 2024].copy()
    train(train_df, sport='nba', model_type='xgb')
    print("NBA model trained.")

def train_mlb():
    model_path = Path('models/mlb_xgb_model.pkl')
    if model_path.exists():
        print("MLB model already exists, skipping")
        return
    print("Training MLB model...")
    from data.mlb_fetcher import build_matchup_features
    from models.trainer import train
    df = build_matchup_features(list(range(2019, 2025)))
    df = df.dropna(subset=['home_win']).reset_index(drop=True)
    train_df = df[df['season'] < 2024].copy()
    train(train_df, sport='mlb', model_type='xgb')
    print("MLB model trained.")

def train_soccer():
    model_path = Path('models/soccer_wc_model.pkl')
    if model_path.exists():
        print("Soccer model already exists, skipping")
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
