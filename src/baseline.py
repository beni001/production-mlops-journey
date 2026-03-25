"""
src/baseline.py — Heuristic lookup table baseline.

Zero ML. No gradients. No parameters. No training.
Just a table of historical averages grouped by hour and day.

Purpose:
  Every ML model must justify its existence by beating this.
  If XGBoost cannot significantly outperform a lookup table,
  XGBoost has no right to be in production. The operational
  complexity of an ML system — versioning, monitoring, retraining —
  is only worth it if the model delivers meaningfully better predictions
  than the simplest possible alternative.

This is the floor. Everything else is measured against it.
"""

import sys
import os
import json
import numpy as np
import pandas as pd

# ── Path setup ──────────────────────────────────────────────────────────────
# The container runs from /app. features/ is at /app/features.
# sys.path tells Python where to look for modules to import.
# Without this line: "ModuleNotFoundError: No module named 'features'"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features.features import load_and_engineer, temporal_split, compute_metrics, TARGET

DATA_PATH    = os.environ.get("DATA_PATH",    "data/raw/rideshare_logs.csv")
METRICS_PATH = os.environ.get("METRICS_PATH", "metrics/baseline_metrics.json")


def build_lookup_table(train: pd.DataFrame) -> pd.DataFrame:
    """
    Build the lookup table from TRAINING data only.

    Groups all training trips by (hour, day_of_week) and computes
    the mean trip_duration for each group.

    This produces a table like:
      hour  day_of_week  mean_duration
      0     0 (Mon)      612s
      0     1 (Tue)      598s
      ...
      17    4 (Fri)      1842s   ← Friday rush hour, highest duration
      ...

    168 possible combinations (24 hours × 7 days).
    The model is literally this table. Nothing more.

    Why build from training data only:
    If you include validation data in the lookup table, the heuristic
    has seen the future — same leakage problem as random splitting.
    The table must be built from the past and applied to the future.
    """
    lookup = (
        train.groupby(["hour", "day_of_week"])[TARGET]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={TARGET: "predicted_duration"})
    )
    print(f"[baseline] Lookup table: {len(lookup)} hour/day combinations")
    print(f"[baseline] Duration range: {lookup['predicted_duration'].min():.0f}s "
          f"— {lookup['predicted_duration'].max():.0f}s")
    return lookup


def predict(val: pd.DataFrame, lookup: pd.DataFrame, global_mean: float) -> np.ndarray:
    """
    Apply the lookup table to validation data.

    For each trip in validation:
      1. Find the matching (hour, day_of_week) row in the lookup table
      2. Use that row's mean_duration as the prediction

    Fallback to global mean if a combination wasn't seen in training.
    Example: if there were no trips at 3am on a Sunday in training data,
    that cell in the lookup table is empty. Global mean fills it.
    In practice this is rare with a full year of NYC taxi data.
    """
    val = val.merge(lookup, on=["hour", "day_of_week"], how="left")

    # Count fallbacks — high fallback rate signals sparse training data
    fallback_count = val["predicted_duration"].isna().sum()
    if fallback_count > 0:
        print(f"[baseline] Fallback to global mean for {fallback_count} rows "
              f"({fallback_count/len(val)*100:.1f}%)")

    val["predicted_duration"] = val["predicted_duration"].fillna(global_mean)
    return val["predicted_duration"].values


def main():
    print("\n[baseline] ═══════════════════════════════════════")
    print("[baseline]  HEURISTIC BASELINE — ZERO ML          ")
    print("[baseline] ═══════════════════════════════════════\n")

    # ── Load and split ───────────────────────────────────────────────────────
    df = load_and_engineer(DATA_PATH)
    train, val = temporal_split(df)

    # ── Build lookup from training data only ─────────────────────────────────
    global_mean = train[TARGET].mean()
    print(f"[baseline] Global mean trip duration: {global_mean:.0f}s "
          f"({global_mean/60:.1f} min)")

    lookup = build_lookup_table(train)

    # ── Predict on validation ─────────────────────────────────────────────────
    predictions = predict(val, lookup, global_mean)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = compute_metrics(val[TARGET], predictions, "heuristic_baseline")

    print("\n[baseline] ── Results ──────────────────────────────")
    print(f"  RMSE  : {metrics['rmse_seconds']}s  ({metrics['rmse_seconds']/60:.1f} min)")
    print(f"  MAPE  : {metrics['mape_pct']}%")
    print(f"  Bias  : {metrics['mean_error_seconds']}s  ", end="")
    if metrics['mean_error_seconds'] > 0:
        print("(over-predicting)")
    elif metrics['mean_error_seconds'] < 0:
        print("(under-predicting)")
    else:
        print("(no bias)")
    print(f"  P95   : {metrics['p95_error_seconds']}s  ({metrics['p95_error_seconds']/60:.1f} min)")
    print(f"  R²    : {metrics['r2']}")
    print("[baseline] ────────────────────────────────────────")
    print("[baseline] This is the floor. ML must beat this.\n")

    # ── Write metrics ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[baseline] Metrics written → {METRICS_PATH}")


if __name__ == "__main__":
    main()
