from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# The Odds API — free tier at https://the-odds-api.com (500 req/month free)
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# Minimum edge (in probability points) to flag a bet
MIN_EDGE_THRESHOLD = 0.03  # 3% edge over implied market probability

# Kelly fraction (1.0 = full Kelly, 0.25 = quarter Kelly — more conservative)
KELLY_FRACTION = 0.25

# Bankroll for backtesting simulation
STARTING_BANKROLL = 1000.0

SPORTS = ["nba", "nfl", "mlb"]
