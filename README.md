# Sports Edge

Statistical edge-finding framework for NBA, NFL, and MLB betting markets.

## Architecture

```
sports-edge/
├── config/settings.py        # thresholds, Kelly fraction, paths
├── data/
│   ├── nba_fetcher.py        # fetch + engineer NBA features
│   ├── nfl_fetcher.py        # (TODO)
│   └── mlb_fetcher.py        # (TODO)
├── models/trainer.py         # time-series CV training + calibration
├── backtesting/
│   ├── backtest.py           # simulate staking + P&L
│   └── line_movement.py      # sharp money / reverse line movement
├── utils/
│   ├── odds.py               # EV, Kelly, vig removal
│   └── features.py           # rolling stats, rest days
└── run_pipeline.py           # main entrypoint
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Add odds API key for real market lines
cp .env.example .env
# edit .env with your key from https://the-odds-api.com

# 3. Run the NBA pipeline
python run_pipeline.py --sport nba --seasons 2019 2020 2021 2022 2023 2024
```

## Key Concepts

**Edge** = model's probability − market's no-vig probability.
A positive edge means the model thinks the true probability is higher than what the market is pricing.

**Expected Value** = `(model_prob × payout) − (1 − model_prob)`
Only bet when EV > 0.

**Kelly Criterion** = optimal bankroll fraction given your edge and payout.
We use quarter-Kelly (0.25×) by default for risk management.

**Calibration** is critical — raw model probabilities are overconfident.
All models use `CalibratedClassifierCV` to produce well-calibrated probabilities.

## Edges Currently Modeled

- Rolling win%, points for/against, point differential (5/10/20 game windows)
- Home/away differentials between teams
- Rest advantage (days since last game)
- Reverse line movement (sharp money signals)
- Public betting bias (overreaction to recency, favorites, home teams)

## Adding NFL / MLB

Follow the same pattern as `data/nba_fetcher.py`:
1. Fetch raw game data (use `nfl_data_py` or `pybaseball`)
2. Build a per-team game log
3. Call `build_matchup_features()` to produce a flat matchup DataFrame
4. Pass to `models/trainer.py` — the trainer is sport-agnostic
