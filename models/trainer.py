"""
Train and evaluate a calibrated probability model for game outcomes.
Works with any sport's matchup DataFrame as long as it has the expected columns.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from xgboost import XGBClassifier

from config.settings import MODELS_DIR

MODELS_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_PATTERNS = [
    "win_pct_diff",
    "point_diff_diff",
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
]


def select_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if any(c.startswith(p) for p in FEATURE_PATTERNS)]


def train(
    df: pd.DataFrame,
    target: str = "home_win",
    sport: str = "nba",
    model_type: str = "xgb",
) -> dict:
    """
    Train a calibrated model using time-series cross-validation.
    Returns a dict with the fitted model, scaler, feature list, and CV metrics.
    """
    df = df.dropna(subset=[target]).copy()
    feature_cols = select_features(df)
    df = df.dropna(subset=feature_cols)

    X = df[feature_cols].values
    y = df[target].values
    dates = pd.to_datetime(df["date"])

    # Time-series split — never train on future data
    tscv = TimeSeriesSplit(n_splits=5)

    metrics = {"brier": [], "auc": [], "log_loss": []}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        if model_type == "logistic":
            base = LogisticRegression(C=1.0, max_iter=1000)
            model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        else:
            base = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                verbosity=0,
            )
            model = CalibratedClassifierCV(base, method="isotonic", cv=3)

        model.fit(X_train_s, y_train)
        probs = model.predict_proba(X_val_s)[:, 1]

        metrics["brier"].append(brier_score_loss(y_val, probs))
        metrics["auc"].append(roc_auc_score(y_val, probs))
        metrics["log_loss"].append(log_loss(y_val, probs))

        print(
            f"  Fold {fold+1}: Brier={metrics['brier'][-1]:.4f} "
            f"AUC={metrics['auc'][-1]:.4f} LogLoss={metrics['log_loss'][-1]:.4f}"
        )

    # Final model trained on all data
    scaler_final = StandardScaler()
    X_final = scaler_final.fit_transform(X)

    if model_type == "logistic":
        final_base = LogisticRegression(C=1.0, max_iter=1000)
        final_model = CalibratedClassifierCV(final_base, method="isotonic", cv=3)
    else:
        final_base = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, verbosity=0,
        )
        final_model = CalibratedClassifierCV(final_base, method="isotonic", cv=3)

    final_model.fit(X_final, y)

    avg_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
    print(f"\nCV averages: {avg_metrics}")

    save_path = MODELS_DIR / f"{sport}_{model_type}_model.pkl"
    joblib.dump({"model": final_model, "scaler": scaler_final, "features": feature_cols}, save_path)
    print(f"Model saved to {save_path}")

    return {
        "model": final_model,
        "scaler": scaler_final,
        "features": feature_cols,
        "metrics": avg_metrics,
    }


def load_model(sport: str = "nba", model_type: str = "xgb") -> dict:
    path = MODELS_DIR / f"{sport}_{model_type}_model.pkl"
    return joblib.load(path)


def predict_proba(model_bundle: dict, df: pd.DataFrame) -> np.ndarray:
    X = df[model_bundle["features"]].values
    X_s = model_bundle["scaler"].transform(X)
    return model_bundle["model"].predict_proba(X_s)[:, 1]
