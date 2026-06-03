import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path

model_path = Path('models/nba_xgb_model.pkl')
if not model_path.exists():
    print('Training model...')
    from data.nba_fetcher import build_matchup_features
    from models.trainer import train
    df = build_matchup_features(list(range(2019, 2025)))
    df = df.dropna(subset=['home_win']).reset_index(drop=True)
    train_df = df[df['season'] < 2024].copy()
    train(train_df, sport='nba', model_type='xgb')
else:
    print('Model already exists, skipping training')
