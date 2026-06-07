"""
Train and evaluate a calibrated probability model for game outcomes.
Works with any sport's matchup DataFrame as long as it has the expected columns.

Calibration approach: train XGBoost on the first 80% of data (chronologically),
then calibrate with Platt sigmoid on the held-out 20%. This prevents the calibration
layer from overfitting to training data, which was causing over-extreme probabilities.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from xgboost import XGBClassifier

from config.settings import MODELS_DIR

MODELS_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_PATTERNS = [
    "win_pct_diff",
    "point_diff_diff",
    "run_diff_diff",
    "rest_advantage",
    "home_win_pct",
    "away_win_pct",
    "home_point_diff",
    "away_point_diff",
    "home_pts_for",
    "away_pts_for",
    "home_pts_against",
    "away_pts_against",
    "home_days_rest",
    "away_days_rest",
    "home_season_win_pct",
    "away_season_win_pct",
    "home_season_run_diff_avg",
    "away_season_run_diff_avg",
    "season_win_pct_diff",
    "season_run_diff_avg_diff",
    "home_starter_era",
    "away_starter_era",
    "home_starter_whip",
    "away_starter_whip",
    "home_starter_k_per_9",
    "away_starter_k_per_9",
    "starter_era_diff",
    "starter_whip_diff",
    "starter_k9_diff",
]


def select_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if any(c.startswith(p) for p in FEATURE_PATTERNS)]


def _print_calibration_diagnostics(y_true: np.ndarray, probs_raw: np.ndarray,
                                   probs_cal: np.ndarray) -> None:
    """Print per-decile calibration table: predicted prob vs actual win rate."""
    print("\n  Calibration diagnostics (predicted % → actual win rate):")
    print(f"  {'Decile':<8} {'Raw pred':>10} {'Cal pred':>10} {'Actual':>10} {'N':>6}")
    print("  " + "-" * 46)
    bins = np.percentile(probs_cal, np.linspace(0, 100, 11))
    bins = np.unique(bins)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs_cal >= lo) & (probs_cal < hi)
        if mask.sum() == 0:
            continue
        actual = y_true[mask].mean()
        raw_mean = probs_raw[mask].mean()
        cal_mean = probs_cal[mask].mean()
        print(f"  {lo:.0%}–{hi:.0%}   {raw_mean:>10.1%} {cal_mean:>10.1%} {actual:>10.1%} {mask.sum():>6}")

    # Overall mean predicted vs actual — should be close to base rate (~54% home wins)
    print(f"\n  Overall: raw mean={probs_raw.mean():.3f}  cal mean={probs_cal.mean():.3f}"
          f"  actual={y_true.mean():.3f}  (base rate = actual home win %)")
    print(f"  Brier (raw)={brier_score_loss(y_true, probs_raw):.4f}"
          f"  Brier (cal)={brier_score_loss(y_true, probs_cal):.4f}")


def train(
    df: pd.DataFrame,
    target: str = "home_win",
    sport: str = "nba",
    model_type: str = "xgb",
) -> dict:
    """
    Train a calibrated model using chronological train/calibration split.

    Strategy:
      1. Sort by date, use first 80% to train XGBoost.
      2. Calibrate with Platt sigmoid on the held-out 20% — this ensures the
         calibration layer never sees training data, preventing over-extreme
         out-of-sample probabilities.
      3. Run time-series CV on the training split to report fold metrics.
    """
    df = df.dropna(subset=[target]).copy()
    feature_cols = select_features(df)
    df = df.dropna(subset=feature_cols)
    df = df.sort_values("date").reset_index(drop=True)

    X = df[feature_cols].values
    y = df[target].values

    # Chronological 80/20 split for calibration
    split = int(len(df) * 0.80)
    X_train_all, X_cal = X[:split], X[split:]
    y_train_all, y_cal = y[:split], y[split:]

    print(f"  Training set: {len(X_train_all)} games | Calibration set: {len(X_cal)} games")

    # ── Time-series CV on the training portion ─────────────────────────────
    tscv = TimeSeriesSplit(n_splits=5)
    metrics = {"brier": [], "auc": [], "log_loss": []}

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train_all)):
        X_tr, X_val = X_train_all[tr_idx], X_train_all[val_idx]
        y_tr, y_val = y_train_all[tr_idx], y_train_all[val_idx]

        scaler = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        if model_type == "logistic":
            base = LogisticRegression(C=1.0, max_iter=1000)
        else:
            base = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", verbosity=0,
            )
        base.fit(X_tr_s, y_tr)
        probs = base.predict_proba(X_val_s)[:, 1]

        metrics["brier"].append(brier_score_loss(y_val, probs))
        metrics["auc"].append(roc_auc_score(y_val, probs))
        metrics["log_loss"].append(log_loss(y_val, probs))
        print(
            f"  Fold {fold+1}: Brier={metrics['brier'][-1]:.4f} "
            f"AUC={metrics['auc'][-1]:.4f} LogLoss={metrics['log_loss'][-1]:.4f}"
        )

    avg_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
    print(f"\n  CV averages (uncalibrated): {avg_metrics}")

    # ── Final base model trained on the full training split ────────────────
    scaler_final = StandardScaler()
    X_train_s = scaler_final.fit_transform(X_train_all)
    X_cal_s   = scaler_final.transform(X_cal)

    if model_type == "logistic":
        base_final = LogisticRegression(C=1.0, max_iter=1000)
    else:
        base_final = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, verbosity=0,
        )
    base_final.fit(X_train_s, y_train_all)

    # ── Calibrate on the held-out 20% using Platt sigmoid ─────────────────
    # cv='prefit' means: base model is already trained, just fit the sigmoid
    # wrapper on the calibration set. Sigmoid (Platt) is more regularized than
    # isotonic and avoids overfitting on moderate-sized calibration sets.
    calibrated_model = CalibratedClassifierCV(base_final, method="sigmoid", cv="prefit")
    calibrated_model.fit(X_cal_s, y_cal)

    # ── Calibration diagnostics on the held-out set ────────────────────────
    probs_raw = base_final.predict_proba(X_cal_s)[:, 1]
    probs_cal = calibrated_model.predict_proba(X_cal_s)[:, 1]
    _print_calibration_diagnostics(y_cal, probs_raw, probs_cal)

    save_path = MODELS_DIR / f"{sport}_{model_type}_model.pkl"
    joblib.dump({
        "model":    calibrated_model,
        "scaler":   scaler_final,
        "features": feature_cols,
    }, save_path)
    print(f"\n  Model saved to {save_path}")

    return {
        "model":    calibrated_model,
        "scaler":   scaler_final,
        "features": feature_cols,
        "metrics":  avg_metrics,
    }


def load_model(sport: str = "nba", model_type: str = "xgb") -> dict:
    path = MODELS_DIR / f"{sport}_{model_type}_model.pkl"
    return joblib.load(path)


def predict_proba(model_bundle: dict, df: pd.DataFrame) -> np.ndarray:
    X = df[model_bundle["features"]].values
    X_s = model_bundle["scaler"].transform(X)
    return model_bundle["model"].predict_proba(X_s)[:, 1]
