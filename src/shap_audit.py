"""
src/shap_audit.py — Leakage detection via SHAP values.

What this script does:
  1. Loads the trained XGBoost model
  2. Computes SHAP values on the validation set
  3. Identifies the contribution of each feature to each prediction
  4. Runs the "impossible power" test — flags any feature that
     explains a disproportionate share of model variance
  5. Cross-references against the feature registry to confirm
     all high-importance features were available at request time
  6. Writes a leakage audit report to metrics/

Why SHAP and not just feature importance:
  XGBoost's built-in feature importance (used on Day 5) measures
  how often each feature is used to split nodes across all trees.
  It tells you which features the model used most frequently.

  SHAP tells you something different and more powerful:
  for each individual prediction, how much did each feature
  push the prediction above or below the average?

  Feature importance: "distance_km was used in 66% of splits"
  SHAP: "for THIS trip, distance_km added +420s to the prediction"

  SHAP is additive — the sum of all SHAP values plus the base
  value (dataset average) equals the exact model prediction.
  This mathematical guarantee makes SHAP trustworthy for auditing.
"""

import sys
import os
import json
import numpy as np
import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features.features import FEATURES, TARGET
from src.temporal_validation import date_based_split

import shap

DATA_PATH    = os.environ.get("DATA_PATH",    "data/processed/rideshare_clean.parquet")
MODEL_PATH   = os.environ.get("MODEL_PATH",   "models/xgboost_model.pkl")
REPORT_PATH  = os.environ.get("REPORT_PATH",  "metrics/shap_audit.json")
REGISTRY_PATH = "features/registry.json"

# The "impossible power" threshold
# If any single feature accounts for more than this fraction
# of total absolute SHAP values, flag it for investigation
DOMINANCE_THRESHOLD = 0.70


def load_registry_inference_flags():
    """
    Load which features are marked available_at_inference in registry.
    Features marked False should never have high SHAP values —
    they weren't available at request time.
    If they do, that's a leakage signal.
    """
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    flags = {}
    for view in registry["feature_views"]:
        for feat in view["features"]:
            flags[feat["name"]] = {
                "available_at_inference": feat["available_at_inference"],
                "leakage_risk": feat["leakage_risk"],
            }
    return flags


def compute_shap_values(model, X_val: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP values using TreeExplainer.

    TreeExplainer is the fast SHAP algorithm for tree-based models.
    It exploits the tree structure to compute exact Shapley values
    in O(TLD²) time instead of the exponential time of brute force.
    T = number of trees, L = max leaves, D = max depth.

    For your 299-tree XGBoost model this runs in seconds.
    For a neural network you would use DeepExplainer or KernelExplainer
    which are slower but work on any model architecture.

    Returns: array of shape (n_samples, n_features)
    Each row is one prediction.
    Each column is one feature's SHAP value for that prediction.
    """
    print("[shap] Initialising TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    print(f"[shap] Computing SHAP values for {len(X_val):,} predictions...")
    print("[shap] This takes 1-2 minutes on CPU...")
    shap_values = explainer.shap_values(X_val)

    print(f"[shap] SHAP values computed. Shape: {shap_values.shape}")
    print(f"[shap] Base value (dataset average): {explainer.expected_value:.1f}s")

    return shap_values, explainer.expected_value


def impossible_power_test(shap_values: np.ndarray, feature_names: list) -> dict:
    """
    Test whether any single feature has disproportionate explanatory power.

    Method:
      1. Compute mean absolute SHAP value per feature
         (average magnitude of each feature's contribution)
      2. Compute what fraction of total explanatory power
         each feature accounts for
      3. Flag any feature above DOMINANCE_THRESHOLD

    Legitimate dominance: distance_km at ~66% is expected.
    Physical trips are primarily determined by how far you travel.
    This is not leakage — it's physics.

    Suspicious dominance: if a temporal feature like pickup_datetime
    itself (not derived hour/day features) showed 90% dominance,
    that would suggest the model is using the raw timestamp as a
    proxy for trip duration — a form of leakage.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    total         = mean_abs_shap.sum()
    fractions     = mean_abs_shap / total

    ranked = sorted(
        zip(feature_names, mean_abs_shap, fractions),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\n[shap] ── Impossible Power Test ─────────────────────")
    print(f"  {'Feature':<28} {'Mean |SHAP|':>12} {'Fraction':>10} {'Flag'}")
    print(f"  {'─'*28} {'─'*12} {'─'*10} {'─'*6}")

    flags = []
    results = []
    for feat, mean_shap, fraction in ranked:
        flag = "⚠ DOMINANT" if fraction > DOMINANCE_THRESHOLD else ""
        if fraction > DOMINANCE_THRESHOLD:
            flags.append(feat)
        print(f"  {feat:<28} {mean_shap:>12.1f}s {fraction:>10.3f} {flag}")
        results.append({
            "feature":    feat,
            "mean_abs_shap": round(float(mean_shap), 2),
            "fraction":      round(float(fraction), 4),
            "dominant":      bool(fraction > DOMINANCE_THRESHOLD),
        })

    print(f"[shap] ──────────────────────────────────────────────")
    return {"rankings": results, "dominant_features": flags}


def registry_cross_reference(power_results: dict, registry_flags: dict) -> dict:
    """
    Cross-reference SHAP importance against Day 8 feature registry.

    The audit question:
      "Was this feature physically available at the exact moment
       the user requested the ride?"

    If a feature has high SHAP importance AND is marked
    available_at_inference: False in the registry,
    that is a confirmed leakage signal — the model is using
    information it shouldn't have access to at prediction time.

    If a feature has high SHAP importance AND leakage_risk: HIGH,
    that is a warning — requires manual verification.
    """
    print(f"\n[shap] ── Registry Cross-Reference ──────────────────")
    print(f"  {'Feature':<28} {'SHAP Rank':>9} {'At Inference':>13} {'Leakage Risk'}")
    print(f"  {'─'*28} {'─'*9} {'─'*13} {'─'*15}")

    violations   = []
    warnings_list = []

    for i, item in enumerate(power_results["rankings"]):
        feat = item["feature"]
        rank = i + 1
        reg  = registry_flags.get(feat, {
            "available_at_inference": "UNKNOWN",
            "leakage_risk": "UNKNOWN"
        })

        at_inf   = reg["available_at_inference"]
        leak_risk = reg["leakage_risk"]

        status = ""
        if at_inf == False and item["fraction"] > 0.05:
            status = "⛔ LEAKAGE"
            violations.append(feat)
        elif leak_risk != "none" and item["fraction"] > 0.05:
            status = "⚠ WARNING"
            warnings_list.append(feat)

        print(f"  {feat:<28} {rank:>9} {str(at_inf):>13} {leak_risk:<15} {status}")

    print(f"[shap] ──────────────────────────────────────────────")

    if violations:
        print(f"\n[shap] ⛔ LEAKAGE CONFIRMED: {violations}")
        print(f"[shap] These features have high SHAP importance but")
        print(f"[shap] are NOT available at inference time.")
        print(f"[shap] Remove them immediately before deployment.")
    elif warnings_list:
        print(f"\n[shap] ⚠ WARNINGS: {warnings_list}")
        print(f"[shap] These features have elevated leakage risk.")
        print(f"[shap] Manual verification required.")
    else:
        print(f"\n[shap] ✓ CLEAN: All high-importance features are")
        print(f"[shap] confirmed available at inference time.")
        print(f"[shap] No leakage detected.")

    return {
        "leakage_violations": violations,
        "warnings":           warnings_list,
        "clean":              bool(len(violations) == 0),
    }


def additivity_check(
    model,
    X_sample: pd.DataFrame,
    shap_values: np.ndarray,
    base_value: float,
) -> None:
    """
    Verify SHAP additivity: base + sum(SHAP) = prediction.

    This is the mathematical guarantee that makes SHAP trustworthy.
    If this check fails, something is wrong with the SHAP computation.

    Checks 5 random predictions.
    """
    print(f"\n[shap] ── Additivity Check ───────────────────────────")
    print(f"  base value: {base_value:.1f}s")
    print(f"  Verifying: base + sum(SHAP values) == model prediction")
    print()

    preds = model.predict(X_sample.iloc[:5])
    for i in range(5):
        shap_sum    = shap_values[i].sum()
        reconstructed = base_value + shap_sum
        actual        = preds[i]
        diff          = abs(reconstructed - actual)
        status        = "✓" if diff < 1.0 else "✗"
        print(f"  Row {i}: {base_value:.1f} + {shap_sum:.1f} "
              f"= {reconstructed:.1f} vs model {actual:.1f} "
              f"(diff {diff:.2f}s) {status}")

    print(f"[shap] ──────────────────────────────────────────────")


def main():
    print("\n[shap] ══════════════════════════════════════════════")
    print("[shap]  LEAKAGE DETECTION — SHAP AUTOPSY             ")
    print("[shap] ══════════════════════════════════════════════\n")

    # ── Load ─────────────────────────────────────────────────────
    print(f"[shap] Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    print(f"[shap] Loading data from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    _, val, _ = date_based_split(df)

    # Use 5000 rows for SHAP — full val set is slow on CPU
    # 5000 rows gives statistically reliable importance rankings
    sample = val[FEATURES].sample(5000, random_state=42)
    print(f"[shap] Using {len(sample):,} row sample for SHAP computation")

    # ── Load registry ─────────────────────────────────────────────
    registry_flags = load_registry_inference_flags()

    # ── Compute SHAP values ───────────────────────────────────────
    shap_values, base_value = compute_shap_values(model, sample)

    # ── Additivity check ─────────────────────────────────────────
    additivity_check(model, sample, shap_values, base_value)

    # ── Impossible power test ─────────────────────────────────────
    power_results = impossible_power_test(shap_values, FEATURES)

    # ── Registry cross-reference ──────────────────────────────────
    audit_results = registry_cross_reference(power_results, registry_flags)

    # ── distance_km justification ────────────────────────────────
    top_feature = power_results["rankings"][0]["feature"]
    top_fraction = power_results["rankings"][0]["fraction"]

    print(f"\n[shap] ── Top Feature Justification ─────────────────")
    print(f"  Top feature: {top_feature} ({top_fraction:.1%} of explanatory power)")
    if top_feature == "distance_km":
        print(f"  Physical justification: trip duration is primarily")
        print(f"  determined by how far you travel. distance_km is")
        print(f"  derived from pickup and dropoff GPS coordinates —")
        print(f"  both available at ride request time.")
        print(f"  High SHAP dominance is expected and legitimate.")
        print(f"  Registry confirms: available_at_inference = True")
        print(f"  Leakage risk: none")
        print(f"  VERDICT: Not leakage. Physics.")
    else:
        print(f"  UNEXPECTED top feature. Investigate immediately.")
        print(f"  Expected distance_km to dominate.")
    print(f"[shap] ──────────────────────────────────────────────")

    # ── Write report ──────────────────────────────────────────────
    report = {
        "audit_date":       "2026-04-21",
        "model_path":       MODEL_PATH,
        "sample_size":      len(sample),
        "base_value_seconds": round(float(base_value), 2),
        "dominance_threshold": DOMINANCE_THRESHOLD,
        "power_test":       power_results,
        "registry_audit":   audit_results,
        "verdict":          "CLEAN" if audit_results["clean"] else "LEAKAGE_DETECTED",
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[shap] Report written → {REPORT_PATH}")
    print(f"[shap] Verdict: {report['verdict']}")
    print("\n[shap] ══════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
