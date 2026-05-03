"""
src/train_weighted.py — Cost-sensitive XGBoost with sample weights.

The problem this solves:
  Standard XGBoost minimises average RMSE across all training rows.
  With 95% short trips and 5% long trips, the model implicitly
  optimises for short trips and systematically under-predicts long ones.

  Baseline slice evaluation revealed:
    Normal trips RMSE:  266s   Bias: -27s
    Spike trips RMSE:   970s   Bias: +584s  ← 10 min systematic under-prediction
    Ratio: 3.65x worse on the trips that matter most commercially

The fix — sample weights:
  XGBoost accepts a sample_weight array during fit().
  Rows with higher weight contribute more to the loss function.
  The model pays proportionally more attention to those rows.

  We assign higher weights to trips in the top 5% of duration.
  The model now "cares more" about getting long trips right,
  at the acceptable cost of slightly worse short trip performance.

Weight scheme:
  Normal trips (duration <= p95): weight = 1.0  (baseline)
  Spike trips  (duration >  p95): weight = 5.0  (5x more important)

Why 5x:
  Spike trips are 5% of data. Without weighting, their total
  contribution to the loss is 5% of the gradient signal.
  At 5x weight, they contribute ~21% of the gradient signal —
  proportional to their business importance, not their frequency.

  The weight is a hyperparameter. 3x, 5x, 10x are all reasonable.
  5x was chosen as a starting point — tune based on slice metrics.
"""

import sys
import os
import json
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features.features import FEATURES, TARGET
from src.temporal_validation import date_based_split
from xgboost import XGBRegressor

DATA_PATH    = os.environ.get("DATA_PATH",    "data/processed/rideshare_clean.parquet")
MODEL_PATH   = os.environ.get("MODEL_PATH",   "models/xgboost_weighted.pkl")
METRICS_PATH = os.environ.get("METRICS_PATH", "metrics/weighted_metrics.json")

# Weight assigned to spike trips during training
SPIKE_WEIGHT = 5.0
# Percentile threshold that defines a spike
SPIKE_PERCENTILE = 0.95


def compute_sample_weights(y_train: np.ndarray, spike_weight: float, percentile: float) -> np.ndarray:
    """
    Assign sample weights based on trip duration.

    Normal trips get weight 1.0.
    Spike trips (top percentile) get weight spike_weight.

    The weight array has the same length as y_train.
    Each row in X_train corresponds to one weight.

    XGBoost uses these weights to scale the gradient contribution
    of each training example. Higher weight = stronger gradient signal
    = model updates more aggressively to reduce error on that row.
    """
    threshold = np.percentile(y_train, percentile * 100)
    weights   = np.where(y_train > threshold, spike_weight, 1.0)

    spike_count  = (y_train > threshold).sum()
    normal_count = len(y_train) - spike_count

    print(f"[weights] Threshold (p{int(percentile*100)}): {threshold:.0f}s ({threshold/60:.1f} min)")
    print(f"[weights] Normal trips: {normal_count:,} rows → weight 1.0")
    print(f"[weights] Spike trips:  {spike_count:,} rows → weight {spike_weight}")
    print(f"[weights] Effective spike data contribution: "
          f"{spike_count*spike_weight / (normal_count + spike_count*spike_weight)*100:.1f}%")

    return weights, threshold


def slice_evaluation(y_true: np.ndarray, y_pred: np.ndarray, threshold: float, label: str) -> dict:
    """
    Evaluate model performance separately on normal and spike trips.

    Global RMSE hides the model's systematic failure on long trips.
    Slice evaluation reveals it.

    Returns metrics for both slices and the ratio between them.
    """
    normal_mask = y_true <= threshold
    spike_mask  = y_true >  threshold

    def slice_metrics(actual, predicted, name):
        errors     = actual - predicted
        abs_errors = np.abs(errors)
        rmse = np.sqrt(np.mean(errors**2))
        bias = np.mean(errors)
        p95  = np.percentile(abs_errors, 95)
        return {
            "slice":        name,
            "rows":         len(actual),
            "rmse_seconds": round(float(rmse), 1),
            "bias_seconds": round(float(bias), 1),
            "p95_seconds":  round(float(p95),  1),
        }

    normal_m = slice_metrics(y_true[normal_mask], y_pred[normal_mask], "normal")
    spike_m  = slice_metrics(y_true[spike_mask],  y_pred[spike_mask],  "spike")
    ratio    = spike_m["rmse_seconds"] / normal_m["rmse_seconds"]

    print(f"\n[{label}] ── Slice Evaluation ──────────────────────────")
    print(f"  Threshold: {threshold:.0f}s ({threshold/60:.1f} min)")
    print(f"\n  Normal trips ({normal_m['rows']:,} rows):")
    print(f"    RMSE: {normal_m['rmse_seconds']}s  "
          f"Bias: {normal_m['bias_seconds']}s  "
          f"P95: {normal_m['p95_seconds']}s")
    print(f"\n  Spike trips ({spike_m['rows']:,} rows):")
    print(f"    RMSE: {spike_m['rmse_seconds']}s  "
          f"Bias: {spike_m['bias_seconds']}s  "
          f"P95: {spike_m['p95_seconds']}s")
    print(f"\n  Spike/Normal RMSE ratio: {ratio:.2f}x")

    if ratio < 2.0:
        print(f"  [EXCELLENT] Ratio < 2x — spike performance well controlled")
    elif ratio < 3.0:
        print(f"  [GOOD]      Ratio 2-3x — acceptable spike handling")
    elif ratio < 4.0:
        print(f"  [MODERATE]  Ratio 3-4x — spike trips still poorly served")
    else:
        print(f"  [POOR]      Ratio > 4x — model systematically fails on spikes")

    print(f"[{label}] ─────────────────────────────────────────────")

    return {
        "label":    label,
        "normal":   normal_m,
        "spike":    spike_m,
        "ratio":    round(ratio, 2),
    }


def main():
    import pandas as pd

    print("\n[weighted] ══════════════════════════════════════════")
    print("[weighted]  COST-SENSITIVE XGBOOST — SPIKE DEFENSE   ")
    print("[weighted] ══════════════════════════════════════════\n")

    # ── Load data ────────────────────────────────────────────────
    print(f"[weighted] Loading from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

    train, val, test = date_based_split(df)

    X_train = train[FEATURES].values
    y_train = train[TARGET].values
    X_val   = val[FEATURES].values
    y_val   = val[TARGET].values

    # ── Compute sample weights ───────────────────────────────────
    print("\n[weighted] Computing sample weights...")
    weights, threshold = compute_sample_weights(
        y_train, SPIKE_WEIGHT, SPIKE_PERCENTILE
    )

    # ── Train weighted model ─────────────────────────────────────
    print("\n[weighted] Training cost-sensitive XGBoost...")
    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        min_child_weight=10,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=20,
        eval_metric="rmse",
    )

    model.fit(
        X_train, y_train,
        sample_weight=weights,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    print(f"\n[weighted] Best iteration: {model.best_iteration}")
    print(f"[weighted] Best val RMSE:  {model.best_score:.1f}s")

    # ── Evaluate ─────────────────────────────────────────────────
    preds = model.predict(X_val)

    baseline_results = slice_evaluation(y_val, preds, threshold, "weighted_model")

    # ── Global metrics ───────────────────────────────────────────
    global_rmse = np.sqrt(np.mean((y_val - preds)**2))
    global_bias = np.mean(y_val - preds)
    global_mape = np.mean(np.abs((y_val - preds) / y_val)) * 100
    global_p95  = np.percentile(np.abs(y_val - preds), 95)
    r2 = 1 - np.sum((y_val-preds)**2) / np.sum((y_val-y_val.mean())**2)

    print(f"\n[weighted] ── Global Metrics ──────────────────────────")
    print(f"  RMSE: {global_rmse:.1f}s")
    print(f"  MAPE: {global_mape:.1f}%")
    print(f"  Bias: {global_bias:.1f}s")
    print(f"  P95:  {global_p95:.1f}s")
    print(f"  R²:   {r2:.4f}")
    print(f"[weighted] ─────────────────────────────────────────────")

    # ── Save ─────────────────────────────────────────────────────
    joblib.dump(model, MODEL_PATH)
    print(f"\n[weighted] Model saved → {MODEL_PATH}")

    metrics = {
        "model":            "xgboost_weighted_v1",
        "spike_weight":     SPIKE_WEIGHT,
        "spike_percentile": SPIKE_PERCENTILE,
        "spike_threshold":  round(float(threshold), 1),
        "global": {
            "rmse_seconds":       round(float(global_rmse), 1),
            "mape_pct":           round(float(global_mape), 1),
            "mean_error_seconds": round(float(global_bias), 1),
            "p95_error_seconds":  round(float(global_p95),  1),
            "r2":                 round(float(r2), 4),
        },
        "slices": baseline_results,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[weighted] Metrics written → {METRICS_PATH}")


if __name__ == "__main__":
    main()
