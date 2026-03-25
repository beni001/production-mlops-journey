"""
src/train_xgboost.py — XGBoost with overfit detection and feature importance.

Purpose:
  Break the linear assumption. Capture non-linear interactions that
  Linear Regression structurally cannot learn.

  Example of what Linear Regression cannot express:
    "A 3km trip at 8am on Friday in Midtown takes 3x longer than
     a 3km trip at 3am on Sunday in Brooklyn."
  Same distance. Completely different duration. The relationship
  between distance and duration depends on hour, day, AND location
  simultaneously. That three-way interaction is non-linear.
  XGBoost learns it. Linear Regression averages over it.

How XGBoost works — the core intuition:
  Round 1: Build one shallow decision tree. It's rough. Residuals are large.
  Round 2: Build a second tree that predicts the RESIDUALS of Round 1.
           Not the target — the errors. It corrects Round 1's mistakes.
  Round 3: Build a third tree correcting Round 1 + Round 2's combined errors.
  ...
  Round N: Final prediction = sum of all N trees' contributions.

  Each tree is a correction mechanism for the ensemble so far.
  This is gradient boosting — you're descending the gradient of
  the loss function one tree at a time.
"""

import sys
import os
import json
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features.features import (
    load_and_engineer, temporal_split,
    compute_metrics, FEATURES, TARGET
)
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

DATA_PATH    = os.environ.get("DATA_PATH",    "data/raw/rideshare_logs.csv")
MODEL_PATH   = os.environ.get("MODEL_PATH",   "models/xgboost_model.pkl")
METRICS_PATH = os.environ.get("METRICS_PATH", "metrics/xgboost_metrics.json")


def build_model() -> XGBRegressor:
    """
    XGBoost with conservative regularization.

    Every parameter below is a deliberate choice against overfitting.
    Overfitting = memorizing training data instead of learning patterns.
    A memorized model performs perfectly on training data and fails
    on anything it hasn't seen — exactly the production failure mode.

    Parameter explanations:

    n_estimators=300
      Number of trees. More trees = more capacity to learn complex patterns.
      Also more capacity to memorize noise.
      early_stopping_rounds handles this — we stop before memorization kicks in.

    max_depth=5
      Maximum depth of each tree. Depth 5 = up to 32 leaf nodes per tree.
      Depth 6 would allow 64 leaves — enough to memorize specific NYC blocks.
      Depth 5 forces the tree to find patterns that generalize across blocks.
      Think of depth as: how specific is this tree allowed to get?

    learning_rate=0.05
      How much each tree contributes to the final ensemble.
      Low learning rate = each tree makes small corrections.
      Requires more trees but each one is more conservative.
      High learning rate = aggressive corrections = overshooting = overfit.
      0.05 is the production-safe default. 0.3 is the "fast but risky" default.

    subsample=0.8
      Each tree is trained on a random 80% of rows.
      The other 20% is different for every tree.
      This means no single tree ever sees the full dataset.
      Trees cannot collude to memorize specific rows.
      Same principle as dropout in neural networks.

    colsample_bytree=0.8
      Each tree uses a random 80% of features.
      Prevents any single feature from dominating every tree.
      Forces the ensemble to discover multiple independent signals.

    reg_lambda=2.0
      L2 regularization on leaf weights.
      Penalises large leaf values — forces the model toward
      smaller, more conservative predictions per tree.
      Higher = smoother model = less overfit.
      Default is 1.0 — we use 2.0 for this spatial dataset because
      location features can drive large leaf values for specific blocks.

    min_child_weight=10
      Minimum number of training samples required to create a new leaf.
      Without this: a tree could create a leaf for a single unusual trip
      and memorize it perfectly.
      With min_child_weight=10: every leaf must represent at least 10 trips.
      Forces generalization over individual data points.

    early_stopping_rounds=20
      If validation RMSE doesn't improve for 20 consecutive rounds, stop.
      This is the most important overfit defense.
      n_estimators=300 is the maximum. Early stopping finds the optimum.
      The model stops at round 87 or 143 or wherever validation peaks —
      not blindly at 300.
    """
    return XGBRegressor(
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


def check_overfit(train_rmse: float, val_rmse: float) -> float:
    """
    Overfit diagnostic.

    A perfect model has identical train and val RMSE.
    A memorizing model has much lower train RMSE than val RMSE —
    it learned the training data specifically, not the underlying pattern.

    The gap percentage measures how much worse the model is on
    data it hasn't seen vs data it trained on.

    Under 15% : healthy generalization
    15-30%    : mild overfit, monitor
    Over 30%  : memorization, increase regularization before deploying

    Why this matters for silent failure:
    A model with 35% overfit gap appears to perform well on your
    validation metrics (which are taken from val RMSE).
    But val RMSE is still optimistic compared to truly unseen
    production data — especially anomalous events.
    The overfit gap is a leading indicator of how badly the model
    will degrade when production data diverges from training data.
    """
    gap_pct = ((val_rmse - train_rmse) / train_rmse) * 100

    print("\n[xgboost] ── Overfit Check ─────────────────────────")
    print(f"  Train RMSE : {train_rmse:.1f}s")
    print(f"  Val RMSE   : {val_rmse:.1f}s")
    print(f"  Gap        : {gap_pct:.1f}%")

    if gap_pct > 30:
        print("  [WARNING] Gap > 30% — model is memorizing training data.")
        print("            Increase reg_lambda, min_child_weight,")
        print("            or reduce max_depth before deploying.")
    elif gap_pct > 15:
        print("  [WATCH]   Gap 15-30% — mild overfit. Monitor in production.")
    else:
        print("  [OK]      Gap < 15% — healthy generalization.")
    print("[xgboost] ─────────────────────────────────────────")

    return gap_pct


def print_feature_importance(model: XGBRegressor) -> None:
    """
    Feature importance — what XGBoost actually used.

    XGBoost tracks how often each feature was used to split nodes
    across all trees, weighted by how much each split improved the
    loss function. This is the 'gain' importance score.

    High importance = this feature is doing real work.
    Near-zero importance = this feature is noise to the model.

    Expected: distance_km and hour near the top.
    Unexpected: if vendor_id is top-ranked, something is wrong —
    vendor should not be a strong predictor of trip duration.

    This is also your first drift detector:
    If you retrain in 6 months and importance rankings change
    dramatically, the underlying data relationships have shifted.
    That shift is concept drift made visible.
    """
    importance = dict(zip(FEATURES, model.feature_importances_))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print("\n[xgboost] ── Feature Importance (gain) ────────────")
    for feat, imp in sorted_imp:
        bar_len = int(imp * 400)
        bar = "█" * min(bar_len, 40)
        print(f"  {feat:<28} {bar:<40} {imp:.4f}")
    print("[xgboost] ─────────────────────────────────────────")


def main():
    print("\n[xgboost] ═══════════════════════════════════════════")
    print("[xgboost]  XGBOOST — GRADIENT BOOSTED ENSEMBLE       ")
    print("[xgboost] ═══════════════════════════════════════════\n")

    # ── Load, engineer, split ────────────────────────────────────────────────
    df = load_and_engineer(DATA_PATH)
    train, val = temporal_split(df)

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_val   = val[FEATURES]
    y_val   = val[TARGET]

    # ── Train ────────────────────────────────────────────────────────────────
    print("[xgboost] Training with early stopping...")
    print("[xgboost] Will stop if val RMSE doesn't improve for 20 rounds.\n")

    model = build_model()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    print(f"\n[xgboost] Stopped at round: {model.best_iteration}")
    print(f"[xgboost] Best val RMSE   : {model.best_score:.2f}s")

    # ── Overfit check ────────────────────────────────────────────────────────
    train_preds = model.predict(X_train)
    val_preds   = model.predict(X_val)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    val_rmse   = np.sqrt(mean_squared_error(y_val,   val_preds))
    gap_pct    = check_overfit(train_rmse, val_rmse)

    # ── Feature importance ───────────────────────────────────────────────────
    print_feature_importance(model)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    metrics = compute_metrics(y_val, val_preds, "xgboost_v1")
    metrics["train_rmse_seconds"] = round(float(train_rmse), 2)
    metrics["overfit_gap_pct"]    = round(float(gap_pct),    2)
    metrics["best_iteration"]     = int(model.best_iteration)

    print("\n[xgboost] ── Results ───────────────────────────────")
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
    print(f"  Trees : {metrics['best_iteration']} (early stopping fired)")
    print("[xgboost] ─────────────────────────────────────────")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH),   exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    print(f"\n[xgboost] Model saved → {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[xgboost] Metrics written → {METRICS_PATH}")


if __name__ == "__main__":
    main()
