"""
src/temporal_validation.py — Strict temporal integrity enforcement.

Day 11 replaces the fraction-based split (75/5/20) with
date-based cutoffs and walk-forward validation.

Why date-based splits are superior to fraction-based:

Fraction-based (Day 5):
  Split the dataset 75/5/20 by row count.
  Reproducible but not interpretable.
  "Training on 75% of rows" tells you nothing about
  which time periods the model has and hasn't seen.
  If the dataset grows, the split boundaries move.

Date-based (Day 11):
  Split by explicit calendar dates.
  "Training on Jan-Apr 2016" is unambiguous.
  Any engineer reading the code knows exactly what
  the model has and hasn't seen.
  Dataset growth doesn't move the boundaries.
  Matches how the system works in production:
  "Retrain on all data before this Monday."

Walk-forward validation:
  A single train/val split can be misleading.
  If your validation month happened to be unusually
  easy (good weather, no events, normal demand),
  RMSE looks great but doesn't reflect real performance.
  Walk-forward tests the model across multiple months,
  revealing whether performance is stable or volatile.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features.features import FEATURES, TARGET, compute_metrics

# ── Date-based split boundaries ──────────────────────────────────
# These are fixed calendar dates — not fractions.
# Changing the dataset size does not move these boundaries.
TRAIN_END   = "2016-04-30"   # last day of training window
GAP_END     = "2016-05-07"   # 1-week gap discarded entirely
VAL_END     = "2016-05-31"   # last day of validation window
# Test set: everything after VAL_END (June 2016)
# Test set is NEVER touched during development.
# It is used exactly once — final honest evaluation.

# ── Walk-forward fold definitions ────────────────────────────────
# Each fold trains on all data up to train_end,
# discards a 1-week gap, validates on the next month.
# Folds are cumulative — each adds one month of training data.
WALK_FORWARD_FOLDS = [
    {
        "name":      "Fold 1 — Jan only",
        "train_end": "2016-01-31",
        "gap_end":   "2016-02-07",
        "val_start": "2016-02-08",
        "val_end":   "2016-02-29",
    },
    {
        "name":      "Fold 2 — Jan-Feb",
        "train_end": "2016-02-29",
        "gap_end":   "2016-03-07",
        "val_start": "2016-03-08",
        "val_end":   "2016-03-31",
    },
    {
        "name":      "Fold 3 — Jan-Mar",
        "train_end": "2016-03-31",
        "gap_end":   "2016-04-07",
        "val_start": "2016-04-08",
        "val_end":   "2016-04-30",
    },
    {
        "name":      "Fold 4 — Jan-Apr (primary)",
        "train_end": "2016-04-30",
        "gap_end":   "2016-05-07",
        "val_start": "2016-05-08",
        "val_end":   "2016-05-31",
    },
]


def date_based_split(
    df: pd.DataFrame,
    train_end: str = TRAIN_END,
    gap_end: str   = GAP_END,
    val_end: str   = VAL_END,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe by explicit calendar dates.

    Returns: train, val, test dataframes.

    The test set (June 2016) is returned but must never be
    used during model development or hyperparameter tuning.
    Use it exactly once for final honest evaluation.

    The gap period is discarded — not returned.
    It exists only to break temporal autocorrelation
    at the train/val boundary.
    """
    df = df.sort_values("pickup_datetime").reset_index(drop=True)

    train = df[df["pickup_datetime"] <= train_end].copy()
    # gap: train_end < pickup_datetime <= gap_end — discarded
    val   = df[
        (df["pickup_datetime"] > gap_end) &
        (df["pickup_datetime"] <= val_end)
    ].copy()
    test  = df[df["pickup_datetime"] > val_end].copy()

    print(f"\n[temporal] ── Date-Based Split ─────────────────────")
    print(f"[temporal] Train : {len(train):>8,} rows  "
          f"({train['pickup_datetime'].min().date()} → "
          f"{train['pickup_datetime'].max().date()})")
    print(f"[temporal] Gap   : discarded   "
          f"({pd.Timestamp(train_end).date()} → "
          f"{pd.Timestamp(gap_end).date()})")
    print(f"[temporal] Val   : {len(val):>8,} rows  "
          f"({val['pickup_datetime'].min().date()} → "
          f"{val['pickup_datetime'].max().date()})")
    print(f"[temporal] Test  : {len(test):>8,} rows  "
          f"({test['pickup_datetime'].min().date()} → "
          f"{test['pickup_datetime'].max().date()})")
    print(f"[temporal] ─────────────────────────────────────────\n")

    # ── Leakage assertion ─────────────────────────────────────────
    # Hard crash if any future data leaked into training.
    # This is the most important assertion in the entire codebase.
    assert val["pickup_datetime"].min() > train["pickup_datetime"].max(), \
        "TEMPORAL LEAKAGE: val data overlaps with train data"
    assert test["pickup_datetime"].min() > val["pickup_datetime"].max(), \
        "TEMPORAL LEAKAGE: test data overlaps with val data"

    return train, val, test


def walk_forward_validation(
    df: pd.DataFrame,
    model_class,
    model_params: dict,
) -> pd.DataFrame:
    """
    Walk-forward (rolling window) validation across 4 folds.

    Why walk-forward instead of a single split:

    A single train/val split gives you one RMSE number.
    That number might be lucky (easy validation month)
    or unlucky (anomalous validation month).
    You cannot tell from one number whether model performance
    is stable or volatile across different time periods.

    Walk-forward gives you 4 RMSE numbers — one per fold.
    If they are consistent (e.g., 320s, 335s, 328s, 333s),
    the model is stable and generalises across months.
    If they are volatile (e.g., 280s, 420s, 310s, 510s),
    the model is sensitive to specific monthly patterns —
    a signal that seasonal effects or concept drift exist.

    Volatility is not a failure. It is information.
    It tells you: "this model needs monthly retraining"
    or "February is structurally different from April."

    Algorithm:
      For each fold:
        1. Filter train rows: pickup_datetime <= fold.train_end
        2. Discard gap rows
        3. Filter val rows: gap_end < pickup_datetime <= val_end
        4. Train model on train rows
        5. Evaluate on val rows
        6. Record metrics
      Compare metrics across folds.
    """
    print("\n[walk_forward] ══════════════════════════════════════")
    print("[walk_forward]  WALK-FORWARD VALIDATION — 4 FOLDS    ")
    print("[walk_forward] ══════════════════════════════════════\n")

    results = []

    for fold in WALK_FORWARD_FOLDS:
        print(f"[walk_forward] {fold['name']}")

        # ── Filter data for this fold ─────────────────────────────
        train = df[
            df["pickup_datetime"] <= fold["train_end"]
        ].copy()

        val = df[
            (df["pickup_datetime"] > fold["gap_end"]) &
            (df["pickup_datetime"] <= fold["val_end"])
        ].copy()

        print(f"  Train: {len(train):,} rows → Val: {len(val):,} rows")

        # ── Leakage check ─────────────────────────────────────────
        assert val["pickup_datetime"].min() > train["pickup_datetime"].max(), \
            f"LEAKAGE in {fold['name']}"

        # ── Train ─────────────────────────────────────────────────
        X_train = train[FEATURES]
        y_train = train[TARGET]
        X_val   = val[FEATURES]
        y_val   = val[TARGET]

        model = model_class(**model_params)

        # Handle XGBoost early stopping
        if hasattr(model, "early_stopping_rounds"):
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)

        # ── Evaluate ──────────────────────────────────────────────
        preds   = model.predict(X_val)
        metrics = compute_metrics(y_val, preds, fold["name"])

        print(f"  RMSE: {metrics['rmse_seconds']:.1f}s  "
              f"MAPE: {metrics['mape_pct']:.1f}%  "
              f"R²: {metrics['r2']:.4f}")

        results.append({
            "fold":       fold["name"],
            "train_rows": len(train),
            "val_rows":   len(val),
            **metrics,
        })

    results_df = pd.DataFrame(results)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n[walk_forward] ── Summary ──────────────────────────")
    print(f"  {'Fold':<28} {'RMSE':>8} {'MAPE%':>7} {'R²':>8}")
    print(f"  {'─'*28} {'─'*8} {'─'*7} {'─'*8}")
    for _, row in results_df.iterrows():
        print(f"  {row['fold']:<28} "
              f"{row['rmse_seconds']:>8.1f} "
              f"{row['mape_pct']:>7.1f} "
              f"{row['r2']:>8.4f}")

    print(f"\n  RMSE mean : {results_df['rmse_seconds'].mean():.1f}s")
    print(f"  RMSE std  : {results_df['rmse_seconds'].std():.1f}s")
    print(f"  RMSE range: {results_df['rmse_seconds'].min():.1f}s "
          f"— {results_df['rmse_seconds'].max():.1f}s")

    # ── Stability verdict ─────────────────────────────────────────
    rmse_cv = results_df["rmse_seconds"].std() / results_df["rmse_seconds"].mean()
    print(f"\n  Coefficient of variation: {rmse_cv:.3f}")
    if rmse_cv < 0.05:
        print("  [STABLE]   CV < 5% — model performance is consistent across months.")
        print("             Monthly retraining is not urgently required.")
    elif rmse_cv < 0.15:
        print("  [MODERATE] CV 5-15% — some monthly variation exists.")
        print("             Monitor for seasonal drift. Retrain quarterly.")
    else:
        print("  [VOLATILE] CV > 15% — significant monthly variation.")
        print("             Model is sensitive to seasonal patterns.")
        print("             Monthly retraining is required.")

    print("[walk_forward] ══════════════════════════════════════\n")

    return results_df


def final_test_evaluation(
    df: pd.DataFrame,
    model,
    label: str = "xgboost_v1"
) -> dict:
    """
    Evaluate on the held-out June 2016 test set.

    This function should be called EXACTLY ONCE — after all
    model development, hyperparameter tuning, and validation
    is complete. Using the test set during development
    constitutes data leakage — you are implicitly tuning
    toward the test set distribution.

    The test set is June 2016. The model has never seen it.
    This is the most honest performance estimate available.
    It answers: "how well does this model predict trips
    in a month it has genuinely never encountered?"
    """
    print("\n[test_eval] ══════════════════════════════════════")
    print("[test_eval]  FINAL TEST EVALUATION — JUNE 2016    ")
    print("[test_eval]  This runs exactly once.               ")
    print("[test_eval] ══════════════════════════════════════\n")

    _, _, test = date_based_split(df)

    X_test = test[FEATURES]
    y_test = test[TARGET]

    preds   = model.predict(X_test)
    metrics = compute_metrics(y_test, preds, label)
    metrics["evaluation_set"] = "june_2016_held_out"
    metrics["test_rows"]      = len(test)

    print(f"[test_eval] Test set: {len(test):,} trips (June 2016)")
    print(f"[test_eval] RMSE : {metrics['rmse_seconds']}s")
    print(f"[test_eval] MAPE : {metrics['mape_pct']}%")
    print(f"[test_eval] Bias : {metrics['mean_error_seconds']}s")
    print(f"[test_eval] P95  : {metrics['p95_error_seconds']}s")
    print(f"[test_eval] R²   : {metrics['r2']}")

    gap = metrics["rmse_seconds"] - 333.49
    print(f"\n[test_eval] vs validation RMSE (333.49s): {gap:+.1f}s")
    if gap > 50:
        print("[test_eval] [WARN] Significant degradation on test set.")
        print("           Model may be overfit to validation period.")
    elif gap > 0:
        print("[test_eval] [OK]  Slight degradation — normal generalisation gap.")
    else:
        print("[test_eval] [OK]  Test performance matches or beats validation.")

    return metrics
