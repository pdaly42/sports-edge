"""
3-class model trainer for soccer (home win / draw / away win).
Separate from trainer.py to keep binary NBA/MLB models clean.

Output: calibrated probabilities for all 3 outcomes.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from config.settings import MODELS_DIR

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Features available in the soccer matchup DataFrame
SOCCER_FEATURE_PATTERNS = [
    "elo_diff",
    "home_elo_pre",
    "away_elo_pre",
    "home_win_pct_10g",
    "away_win_pct_10g",
    "home_gf_avg_10g",
    "away_gf_avg_10g",
    "home_ga_avg_10g",
    "away_ga_avg_10g",
    "home_gd_avg_10g",
    "away_gd_avg_10g",
    "win_pct_diff_10g",
    "gd_diff_10g",
]


def select_features(df: pd.DataFrame) -> list:
    return [c for c in df.columns if any(c.startswith(p) for p in SOCCER_FEATURE_PATTERNS)]


def train_soccer(df: pd.DataFrame, sport: str = "soccer_wc") -> dict:
    """
    Train a 3-class XGBoost model.
    target column: result_class (0=away win, 1=draw, 2=home win)
    Returns bundle dict with model, scaler, feature list, and CV metrics.
    """
    target = "result_class"
    df = df.dropna(subset=[target]).copy()
    feature_cols = select_features(df)
    df = df.dropna(subset=feature_cols)

    df = df.sort_values("date").reset_index(drop=True) if "date" in df.columns else df

    print(f"  Training on {len(df)} rows | {len(feature_cols)} features")
    print(f"  Class distribution: {df[target].value_counts().sort_index().to_dict()}")

    X = df[feature_cols].values
    y = df[target].values.astype(int)

    # Chronological 80/20 split — calibrate on held-out data only
    split = int(len(df) * 0.80)
    X_train_all, X_cal = X[:split], X[split:]
    y_train_all, y_cal = y[:split], y[split:]
    print(f"  Training split: {len(X_train_all)} | Calibration split: {len(X_cal)}")

    tscv = TimeSeriesSplit(n_splits=5)
    fold_logloss = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_all)):
        X_train, X_val = X_train_all[train_idx], X_train_all[val_idx]
        y_train, y_val = y_train_all[train_idx], y_train_all[val_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)

        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=400,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=1,
            eval_metric="mlogloss",
            verbosity=0,
        )
        model.fit(X_train_s, y_train)
        probs = model.predict_proba(X_val_s)
        ll = log_loss(y_val, probs)
        fold_logloss.append(ll)

        preds = np.argmax(probs, axis=1)
        acc = (preds == y_val).mean()
        print(f"  Fold {fold+1}: LogLoss={ll:.4f}  Accuracy={acc:.3f}")

    avg_ll = float(np.mean(fold_logloss))
    print(f"\n  CV avg LogLoss (uncalibrated): {avg_ll:.4f}")

    # ── Base model on full training split ──────────────────────────────────
    scaler_final = StandardScaler()
    X_train_s = scaler_final.fit_transform(X_train_all)
    X_cal_s   = scaler_final.transform(X_cal)

    base_model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1,
        verbosity=0,
    )
    base_model.fit(X_train_s, y_train_all)

    # ── Calibrate on the held-out 20% ──────────────────────────────────────
    # sklearn CalibratedClassifierCV supports multiclass with sigmoid (one-vs-rest)
    from sklearn.calibration import CalibratedClassifierCV
    calibrated_model = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
    calibrated_model.fit(X_cal_s, y_cal)

    # Calibration diagnostics
    probs_raw = base_model.predict_proba(X_cal_s)
    probs_cal = calibrated_model.predict_proba(X_cal_s)
    ll_raw = log_loss(y_cal, probs_raw)
    ll_cal = log_loss(y_cal, probs_cal)
    print(f"\n  Calibration holdout — LogLoss raw={ll_raw:.4f}  cal={ll_cal:.4f}")
    labels = ["away_win", "draw", "home_win"]
    for i, label in enumerate(labels):
        print(f"    {label}: mean raw={probs_raw[:,i].mean():.3f}  "
              f"cal={probs_cal[:,i].mean():.3f}  "
              f"actual={( y_cal == i).mean():.3f}")

    # Feature importance
    importances = pd.Series(
        base_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print("\n  Top 10 features:")
    for feat, imp in importances.head(10).items():
        print(f"    {feat:<35} {imp:.4f}")

    save_path = MODELS_DIR / f"{sport}_model.pkl"
    bundle = {
        "model":    calibrated_model,
        "scaler":   scaler_final,
        "features": feature_cols,
        "metrics":  {"log_loss": avg_ll},
        "num_class": 3,
    }
    joblib.dump(bundle, save_path)
    print(f"\n  Model saved → {save_path}")
    return bundle


def load_soccer_model(sport: str = "soccer_wc") -> dict:
    path = MODELS_DIR / f"{sport}_model.pkl"
    return joblib.load(path)


def predict_soccer_proba(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """
    Returns array of shape (n_games, 3):
      col 0 = P(away win), col 1 = P(draw), col 2 = P(home win)
    """
    feat_cols = bundle["features"]
    aligned = pd.DataFrame(columns=feat_cols)
    for c in feat_cols:
        aligned[c] = df[c].values if c in df.columns else 0.0
    X = bundle["scaler"].transform(aligned.values)
    return bundle["model"].predict_proba(X)
