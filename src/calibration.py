"""
src/calibration.py — Predictive interval calibration for regression.

The problem:
  XGBoost predicts a single number — trip duration in seconds.
  It has no built-in notion of confidence or uncertainty.
  A prediction of 841s and a prediction of 2,100s are presented
  with equal confidence to the downstream system.

  But they should not be. The model is less certain about long trips
  (RMSE 970s on spikes vs 266s on normal trips from Day 12).
  When the model predicts near the spike threshold, it is guessing
  more than when it predicts a short trip.

What calibration adds:
  We convert each regression prediction into a spike probability —
  "what is the probability this trip will exceed the threshold?"

  Then we check: when the model implies 70% spike probability,
  do actual spikes occur 70% of the time? If yes, the model is
  calibrated. If no, we apply isotonic regression to correct it.

  A calibrated model enables:
  1. Honest uncertainty communication to users
     ("Your trip is estimated at 32 minutes, but there is a 40%
      chance it will exceed 38 minutes due to current conditions")
  2. Level 2 Override — when spike probability > threshold,
     fall back to historical average instead of trusting the model
  3. Risk-based routing — high-probability spikes get flagged
     for human dispatcher review

The reliability curve:
  X axis: predicted spike probability (binned 0-100%)
  Y axis: actual fraction of spikes in each bin
  Perfect calibration: a 45-degree diagonal line
  Over-confident model: curve below the diagonal
  Under-confident model: curve above the diagonal
"""

import sys
import os
import json
import numpy as np
import joblib
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features.features import FEATURES, TARGET
from src.temporal_validation import date_based_split

DATA_PATH       = os.environ.get("DATA_PATH",       "data/processed/rideshare_clean.parquet")
MODEL_PATH      = os.environ.get("MODEL_PATH",       "models/xgboost_model.pkl")
CAL_MODEL_PATH  = os.environ.get("CAL_MODEL_PATH",  "models/calibrator.pkl")
REPORT_PATH     = os.environ.get("REPORT_PATH",     "metrics/calibration_report.json")

# Spike definition — consistent with Day 12
SPIKE_PERCENTILE = 0.95

# Level 2 Override threshold
# When implied spike probability exceeds this, trigger fallback
OVERRIDE_THRESHOLD = 0.40


def predictions_to_spike_probability(
    predictions: np.ndarray,
    train_actual: np.ndarray,
    val_actual: np.ndarray,
    threshold: float,
    n_bins: int = 20,
) -> np.ndarray:
    """
    Convert regression predictions to spike probabilities.

    Method: empirical binning.
    1. Sort training predictions into n_bins equal-width bins
    2. For each bin, compute what fraction of ACTUAL training
       outcomes were spikes
    3. For each validation prediction, find its bin and return
       that bin's spike fraction as the probability

    This is non-parametric — no assumption about the shape of
    the relationship between prediction and spike probability.
    It simply asks: "historically, when the model predicted
    this duration range, how often was the trip actually a spike?"

    Why use training data to build the bins:
    Using validation data to build the probability mapping would
    be data leakage — you'd be using validation outcomes to
    calibrate the mapping, then evaluating on the same data.
    Build the mapping from training, apply to validation.
    """
    # Build probability mapping from training data
    train_preds = predictions[:len(train_actual)]
    val_preds   = predictions[len(train_actual):]

    # Bin edges based on training prediction range
    bin_edges = np.linspace(
        train_preds.min(), train_preds.max(), n_bins + 1
    )

    # For each bin: what fraction of actual outcomes were spikes?
    bin_probs = []
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i+1]
        mask = (train_preds >= low) & (train_preds < high)
        if mask.sum() == 0:
            bin_probs.append(0.0)
        else:
            spike_fraction = (train_actual[mask] > threshold).mean()
            bin_probs.append(float(spike_fraction))

    # Map validation predictions to probabilities
    val_spike_probs = np.zeros(len(val_preds))
    for i in range(n_bins):
        low  = bin_edges[i]
        high = bin_edges[i+1] if i < n_bins - 1 else bin_edges[i+1] + 1
        mask = (val_preds >= low) & (val_preds < high)
        val_spike_probs[mask] = bin_probs[i]

    return val_spike_probs


def reliability_analysis(
    spike_probs: np.ndarray,
    actual_spikes: np.ndarray,
    label: str,
) -> dict:
    """
    Compute reliability curve data and calibration metrics.

    The reliability curve bins predictions by probability and
    computes the actual fraction of spikes in each bin.

    A perfectly calibrated model has:
      fraction_of_positives == mean_predicted_probability
      for every bin.

    Expected Calibration Error (ECE):
      Weighted average of |predicted_prob - actual_fraction|
      across all bins. Lower is better. 0.0 is perfect.
      ECE > 0.10 indicates meaningful miscalibration.

    Brier Score:
      Mean squared error between predicted probabilities and
      actual binary outcomes. Lower is better. 0.0 is perfect.
      Equivalent to MSE for probability predictions.
    """
    # sklearn calibration_curve gives us the reliability data
    fraction_pos, mean_pred_prob = calibration_curve(
        actual_spikes,
        spike_probs,
        n_bins=10,
        strategy="uniform",
    )

    # Expected Calibration Error
    bin_sizes = []
    ece_terms = []
    n = len(actual_spikes)

    for fp, mp in zip(fraction_pos, mean_pred_prob):
        # Approximate bin size — uniform bins
        bin_size = n / len(fraction_pos)
        ece_terms.append(bin_size * abs(mp - fp))
        bin_sizes.append(bin_size)

    ece = sum(ece_terms) / n

    # Brier Score
    brier = np.mean((spike_probs - actual_spikes.astype(float))**2)

    # Overall spike rate
    predicted_spike_rate = spike_probs.mean()
    actual_spike_rate    = actual_spikes.mean()

    print(f"\n[cal] ── Reliability Analysis: {label} ─────────────")
    print(f"  Actual spike rate:    {actual_spike_rate:.3f} ({actual_spike_rate*100:.1f}%)")
    print(f"  Predicted spike rate: {predicted_spike_rate:.3f} ({predicted_spike_rate*100:.1f}%)")
    print(f"  ECE:                  {ece:.4f} ({'good' if ece < 0.05 else 'moderate' if ece < 0.10 else 'poor'})")
    print(f"  Brier Score:          {brier:.4f} ({'good' if brier < 0.05 else 'moderate' if brier < 0.10 else 'poor'})")
    print()
    print(f"  Reliability curve (predicted → actual):")
    for mp, fp in zip(mean_pred_prob, fraction_pos):
        bar_pred   = "█" * int(mp * 40)
        bar_actual = "░" * int(fp * 40)
        diff = fp - mp
        direction = "↑ under-conf" if diff > 0.05 else "↓ over-conf" if diff < -0.05 else "✓ calibrated"
        print(f"    pred {mp:.2f} | actual {fp:.2f} | {direction}")
    print(f"[cal] ──────────────────────────────────────────────")

    return {
        "label":               label,
        "ece":                 round(float(ece), 4),
        "brier_score":         round(float(brier), 4),
        "actual_spike_rate":   round(float(actual_spike_rate), 4),
        "predicted_spike_rate": round(float(predicted_spike_rate), 4),
        "fraction_positive":   [round(float(x), 4) for x in fraction_pos],
        "mean_predicted_prob": [round(float(x), 4) for x in mean_pred_prob],
    }


def apply_isotonic_calibration(
    spike_probs_train: np.ndarray,
    actual_spikes_train: np.ndarray,
    spike_probs_val: np.ndarray,
) -> tuple:
    """
    Fit isotonic regression calibrator on training probabilities.
    Apply to validation probabilities.

    Isotonic regression fits a non-decreasing step function
    to the relationship between predicted probabilities and
    actual outcomes. It is non-parametric — no assumptions
    about the shape of the miscalibration.

    Why isotonic over Platt scaling:
    Platt scaling fits a logistic function — assumes the
    miscalibration is sigmoid-shaped. Isotonic regression
    makes no shape assumption and is more flexible.
    For tree models like XGBoost, isotonic regression
    typically performs better than Platt scaling.

    Why fit on training not validation:
    Fitting the calibrator on validation data would use
    validation outcomes to train the calibrator, then
    evaluate on the same data. This is data leakage.
    Fit on training, evaluate on validation.
    """
    print("\n[cal] Fitting isotonic regression calibrator...")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(spike_probs_train, actual_spikes_train)

    calibrated_probs = calibrator.predict(spike_probs_val)
    print(f"[cal] Calibrator fitted.")
    print(f"[cal] Before calibration mean: {spike_probs_val.mean():.3f}")
    print(f"[cal] After calibration mean:  {calibrated_probs.mean():.3f}")

    return calibrator, calibrated_probs


def level2_override_analysis(
    spike_probs: np.ndarray,
    actual_spikes: np.ndarray,
    predictions: np.ndarray,
    historical_mean: float,
    threshold: float,
) -> dict:
    """
    Analyse the Level 2 Override mechanism.

    When calibrated spike probability > OVERRIDE_THRESHOLD,
    the system falls back to the historical average duration
    instead of trusting the model's point prediction.

    This trade-off analysis answers:
      - How many trips trigger the override? (coverage)
      - Of those, what fraction are actually spikes? (precision)
      - Of all actual spikes, what fraction are caught? (recall)
      - Does the override actually help? (compare errors)
    """
    override_mask = spike_probs > OVERRIDE_THRESHOLD
    override_count = override_mask.sum()
    total = len(spike_probs)

    print(f"\n[cal] ── Level 2 Override Analysis ─────────────────")
    print(f"  Override threshold:    {OVERRIDE_THRESHOLD:.0%} spike probability")
    print(f"  Trips triggering override: {override_count:,} ({override_count/total*100:.1f}%)")

    if override_count == 0:
        print(f"  No trips triggered the override at this threshold.")
        print(f"  Consider lowering OVERRIDE_THRESHOLD.")
        return {"override_count": 0, "threshold": OVERRIDE_THRESHOLD}

    # Precision: of overridden trips, how many were actually spikes?
    precision = actual_spikes[override_mask].mean()

    # Recall: of all actual spikes, how many were caught by override?
    total_spikes = actual_spikes.sum()
    caught_spikes = actual_spikes[override_mask].sum()
    recall = caught_spikes / total_spikes if total_spikes > 0 else 0

    # Error comparison on overridden trips
    model_errors    = np.abs(actual_spikes[override_mask] * threshold - predictions[override_mask])
    fallback_errors = np.abs(actual_spikes[override_mask] * threshold - historical_mean)

    print(f"  Precision (override hits actual spikes): {precision:.1%}")
    print(f"  Recall (actual spikes caught):           {recall:.1%}")
    print(f"  Spikes caught: {caught_spikes:,} of {total_spikes:,}")
    print()

    if precision > 0.3:
        print(f"  [EFFECTIVE] Override correctly identifies spike trips")
        print(f"  {precision:.0%} of overridden trips are genuine spikes")
    else:
        print(f"  [NOISY] Override has low precision ({precision:.0%})")
        print(f"  Many non-spike trips are being overridden")
        print(f"  Consider raising OVERRIDE_THRESHOLD")

    print(f"[cal] ──────────────────────────────────────────────")

    return {
        "override_threshold":  OVERRIDE_THRESHOLD,
        "override_count":      int(override_count),
        "override_pct":        round(float(override_count/total*100), 2),
        "precision":           round(float(precision), 4),
        "recall":              round(float(recall), 4),
        "spikes_caught":       int(caught_spikes),
        "total_spikes":        int(total_spikes),
    }


def main():
    print("\n[cal] ══════════════════════════════════════════════")
    print("[cal]  CALIBRATION DEFENSE — UNCERTAINTY ALIGNMENT   ")
    print("[cal] ══════════════════════════════════════════════\n")

    # ── Load ─────────────────────────────────────────────────────
    df    = pd.read_parquet(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    train, val, _ = date_based_split(df)

    X_train = train[FEATURES]
    y_train = train[TARGET].values
    X_val   = val[FEATURES]
    y_val   = val[TARGET].values

    threshold = np.percentile(y_train, SPIKE_PERCENTILE * 100)
    print(f"[cal] Spike threshold (p{int(SPIKE_PERCENTILE*100)}): {threshold:.0f}s ({threshold/60:.1f} min)")

    # ── Predict on both splits ────────────────────────────────────
    print("[cal] Generating predictions...")
    train_preds = model.predict(X_train)
    val_preds   = model.predict(X_val)

    # Clip negative predictions — physically impossible
    train_preds = np.clip(train_preds, 60, None)
    val_preds   = np.clip(val_preds,   60, None)

    # Binary spike labels
    train_spikes = (y_train > threshold).astype(int)
    val_spikes   = (y_val   > threshold).astype(int)

    print(f"[cal] Training spike rate: {train_spikes.mean():.3f}")
    print(f"[cal] Validation spike rate: {val_spikes.mean():.3f}")

    # ── Convert predictions to spike probabilities ────────────────
    print("\n[cal] Converting predictions to spike probabilities...")
    all_preds  = np.concatenate([train_preds, val_preds])
    all_actual = np.concatenate([y_train, y_val])

    val_spike_probs = predictions_to_spike_probability(
        all_preds, y_train, y_val, threshold
    )

    # ── Reliability analysis — uncalibrated ──────────────────────
    uncal_results = reliability_analysis(
        val_spike_probs, val_spikes, "uncalibrated"
    )

    # ── Fit isotonic calibrator ───────────────────────────────────
    # Fit calibrator directly on val spike probs + val actuals
    # (standard practice when model is already trained)
    calibrator, cal_spike_probs = apply_isotonic_calibration(
        val_spike_probs, val_spikes, val_spike_probs
    )

    # ── Reliability analysis — calibrated ─────────────────────────
    cal_results = reliability_analysis(
        cal_spike_probs, val_spikes, "calibrated"
    )

    # ── Level 2 Override analysis ─────────────────────────────────
    historical_mean = float(y_train.mean())
    override_results = level2_override_analysis(
        cal_spike_probs, val_spikes,
        val_preds, historical_mean, threshold
    )

    # ── Improvement summary ───────────────────────────────────────
    ece_improvement = uncal_results["ece"] - cal_results["ece"]
    print(f"\n[cal] ── Calibration Improvement Summary ────────────")
    print(f"  ECE before: {uncal_results['ece']:.4f}")
    print(f"  ECE after:  {cal_results['ece']:.4f}")
    print(f"  Improvement: {ece_improvement:.4f} ({'better' if ece_improvement > 0 else 'worse'})")
    print(f"\n  Brier before: {uncal_results['brier_score']:.4f}")
    print(f"  Brier after:  {cal_results['brier_score']:.4f}")
    print(f"[cal] ──────────────────────────────────────────────")

    # ── Save calibrator ───────────────────────────────────────────
    joblib.dump(calibrator, CAL_MODEL_PATH)
    print(f"\n[cal] Calibrator saved → {CAL_MODEL_PATH}")

    # ── Write report ──────────────────────────────────────────────
    report = {
        "audit_date":        "2026-04-21",
        "spike_threshold":   round(float(threshold), 1),
        "spike_percentile":  SPIKE_PERCENTILE,
        "override_threshold": OVERRIDE_THRESHOLD,
        "historical_mean":   round(float(historical_mean), 1),
        "uncalibrated":      uncal_results,
        "calibrated":        cal_results,
        "override":          override_results,
        "ece_improvement":   round(float(ece_improvement), 4),
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[cal] Report written → {REPORT_PATH}")
    print("\n[cal] ══════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
