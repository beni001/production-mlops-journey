"""
src/train_linear.py — Linear Regression with coefficient inspection.

Purpose:
  Establish the linear assumption floor.
  Answer the question: how much of trip duration can be explained
  by a straight-line relationship between features and target?

  If XGBoost barely beats this, the problem is mostly linear and
  XGBoost's complexity is hard to justify.
  If XGBoost destroys this, the problem has non-linear structure
  that justifies the ensemble complexity.

  The coefficients are this model's most valuable output —
  they tell you exactly what the model learned in plain numbers.
  XGBoost cannot do this. That interpretability is Linear
  Regression's core advantage over every complex model.
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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_PATH    = os.environ.get("DATA_PATH",          "data/raw/rideshare_logs.csv")
MODEL_PATH   = os.environ.get("LINEAR_MODEL_PATH",  "models/linear_model.pkl")
METRICS_PATH = os.environ.get("METRICS_PATH",       "metrics/linear_metrics.json")


def build_pipeline() -> Pipeline:
    """
    StandardScaler + LinearRegression wrapped in a sklearn Pipeline.

    Why StandardScaler before Linear Regression:

    Your features live on wildly different scales:
      distance_km     : 0.1 — 50
      pickup_latitude : 40.6 — 40.9
      hour            : 0 — 23
      is_weekend      : 0 or 1
      passenger_count : 1 — 6

    Linear Regression fits one coefficient per feature.
    Without scaling, the raw magnitude of a feature influences
    how large its coefficient appears — not its actual importance.
    distance_km ranging 0-50 will get a smaller coefficient than
    is_weekend ranging 0-1, even if distance matters far more.

    StandardScaler transforms every feature to mean=0, std=1.
    After scaling, all features speak the same language.
    Coefficients become comparable — the largest absolute value
    genuinely indicates the most influential feature.

    Why Pipeline:
    A Pipeline chains transformations into one object.
    pipeline.fit(X_train) fits the scaler on training data only,
    then transforms it.
    pipeline.predict(X_val) applies the SAME scaler fitted on
    training data to validation data — never re-fits on val.

    Without Pipeline, the most common bug:
      scaler.fit_transform(X_train)  ← correct
      scaler.fit_transform(X_val)    ← WRONG — refits on val data
                                        leaks val statistics into scaling
    Pipeline makes this bug structurally impossible.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression(n_jobs=-1))
    ])


def inspect_coefficients(pipeline: Pipeline) -> None:
    """
    Print coefficients sorted by absolute magnitude.

    Each coefficient answers:
      "Holding all other features constant, what does one
       standard deviation increase in this feature do to
       predicted trip duration in seconds?"

    After StandardScaler, one unit = one standard deviation.
    So coefficients are directly comparable across features.

    Expected relationships (sanity checks):
      distance_km     : large positive  — longer trip = longer duration
      is_rush_hour    : positive        — rush hour trips take longer
      is_weekend      : negative        — weekends have less traffic
      hour (nonlinear): weak            — hour alone is too coarse
                                          XGBoost will handle this better

    If distance_km is NOT the strongest positive predictor,
    something is wrong — either with the data or the features.
    That's a signal to investigate before trusting any model.
    """
    scaler = pipeline.named_steps["scaler"]
    model  = pipeline.named_steps["model"]

    # Pair each feature name with its coefficient
    coef_pairs = sorted(
        zip(FEATURES, model.coef_),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    print("\n[linear] ── Coefficients (scaled units → seconds) ──")
    print(f"  {'Feature':<28} {'Coefficient':>12}  {'Direction'}")
    print(f"  {'─'*28} {'─'*12}  {'─'*15}")

    for feat, coef in coef_pairs:
        direction = "longer trip  ↑" if coef > 0 else "shorter trip ↓"
        print(f"  {feat:<28} {coef:>+12.1f}s  {direction}")

    print()

    # Sanity check — distance must be a strong positive predictor
    coef_dict = dict(zip(FEATURES, model.coef_))
    if coef_dict["distance_km"] < 0:
        print("  [WARNING] distance_km has a NEGATIVE coefficient.")
        print("            Longer distance predicting shorter duration")
        print("            is physically impossible. Check the data.")
    else:
        print(f"  [OK] distance_km is positive ({coef_dict['distance_km']:+.1f}s)")
        print(f"       Longer trips take longer. Physics checks out.")


def main():
    print("\n[linear] ═══════════════════════════════════════════")
    print("[linear]  LINEAR REGRESSION — LINEAR ASSUMPTION FLOOR")
    print("[linear] ═══════════════════════════════════════════\n")

    # ── Load, engineer, split ────────────────────────────────────────────────
    df = load_and_engineer(DATA_PATH)
    train, val = temporal_split(df)

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_val   = val[FEATURES]
    y_val   = val[TARGET]

    # ── Train ────────────────────────────────────────────────────────────────
    print("[linear] Fitting StandardScaler + LinearRegression...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("[linear] Training complete.")

    # ── Coefficient inspection ───────────────────────────────────────────────
    inspect_coefficients(pipeline)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    predictions = pipeline.predict(X_val)
    # Clip negative predictions — Linear Regression can predict
    # negative durations for very short trips. Physically impossible.
    # Clip to 60s minimum (our data quality floor from features.py)
    predictions = np.clip(predictions, 60, None)

    metrics = compute_metrics(y_val, predictions, "linear_regression")

    print("[linear] ── Results ────────────────────────────────")
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
    print("[linear] ───────────────────────────────────────────")

    # ── What Linear Regression's R² tells you ───────────────────────────────
    r2 = metrics["r2"]
    unexplained = round((1 - r2) * 100, 1)
    print(f"\n[linear] R²={r2} means {round(r2*100,1)}% of trip duration")
    print(f"[linear] variance is explained by linear relationships.")
    print(f"[linear] {unexplained}% is non-linear structure XGBoost")
    print(f"[linear] can potentially capture.\n")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH),   exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"[linear] Model saved → {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[linear] Metrics written → {METRICS_PATH}")


if __name__ == "__main__":
    main()
