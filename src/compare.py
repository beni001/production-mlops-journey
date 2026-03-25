"""
src/compare.py — Side-by-side five-metric comparison of all three models.

Purpose:
  Single source of truth for Day 5 results.
  Reads the three metrics JSON files written by baseline.py,
  train_linear.py, and train_xgboost.py and produces a formatted
  comparison table with verdicts.

  This script has no ML logic. It only reads and interprets results.
  That separation is intentional — evaluation is independent of training.
"""

import json
import os
import sys

METRICS = {
    "Heuristic Baseline": "metrics/baseline_metrics.json",
    "Linear Regression":  "metrics/linear_metrics.json",
    "XGBoost":            "metrics/xgboost_metrics.json",
}


def load_metrics() -> dict:
    results = {}
    missing = []

    for name, path in METRICS.items():
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
        else:
            missing.append(name)

    if missing:
        print(f"[compare] Missing metrics for: {', '.join(missing)}")
        print(f"[compare] Run the missing scripts first.")
        if len(missing) == len(METRICS):
            sys.exit(1)

    return results


def print_comparison_table(results: dict) -> None:
    """
    Five-metric comparison table.

    Column explanations printed below the table so the output
    is self-documenting — anyone reading the terminal output
    understands what they're looking at without needing the code.
    """
    col_w = 24

    print("\n" + "═" * 95)
    print("  DAY 5 — MODEL COMPARISON")
    print("  All models trained on identical features, identical temporal split.")
    print("  Differences in metrics reflect model capacity only.")
    print("═" * 95)

    # Header
    print(f"\n  {'Model':<{col_w}} {'RMSE(s)':>9} {'MAPE%':>7} {'Bias(s)':>9} {'P95(s)':>9} {'R²':>7}")
    print(f"  {'─'*col_w} {'─'*9} {'─'*7} {'─'*9} {'─'*9} {'─'*7}")

    for name, m in results.items():
        bias_sign = "+" if m["mean_error_seconds"] > 0 else ""
        print(
            f"  {name:<{col_w}} "
            f"{m['rmse_seconds']:>9.1f} "
            f"{m['mape_pct']:>7.1f} "
            f"{bias_sign}{m['mean_error_seconds']:>8.1f} "
            f"{m['p95_error_seconds']:>9.1f} "
            f"{m['r2']:>7.4f}"
        )

    print(f"\n  {'─'*col_w} {'─'*9} {'─'*7} {'─'*9} {'─'*9} {'─'*7}")
    print("  RMSE    : lower is better. Root mean squared error in seconds.")
    print("  MAPE%   : lower is better. Mean absolute % error. Scale-invariant.")
    print("  Bias(s) : closest to 0 is best. + = over-predicting. - = under-predicting.")
    print("  P95(s)  : lower is better. Your worst 5% of predictions.")
    print("  R²      : higher is better. 1.0 = perfect. 0.0 = predicts the mean.")


def print_verdicts(results: dict) -> None:
    print("\n" + "═" * 95)
    print("  VERDICTS")
    print("═" * 95)

    baseline = results.get("Heuristic Baseline")
    linear   = results.get("Linear Regression")
    xgboost  = results.get("XGBoost")

    # ── XGBoost vs Heuristic ─────────────────────────────────────────────────
    if baseline and xgboost:
        improvement = (
            (baseline["rmse_seconds"] - xgboost["rmse_seconds"])
            / baseline["rmse_seconds"] * 100
        )
        print(f"\n  XGBoost vs Heuristic Baseline:")
        print(f"    RMSE improvement: {improvement:.1f}%")

        if improvement < 0:
            print("    [FAIL] XGBoost is WORSE than a lookup table.")
            print("           Do not deploy. Investigate features and data quality.")
        elif improvement < 10:
            print("    [WEAK] Under 10% improvement over a lookup table.")
            print("           XGBoost complexity is hard to justify.")
            print("           Consider: more features, larger dataset, different model.")
        elif improvement < 25:
            print("    [OK]   Moderate improvement. Complexity has a cost.")
            print("           Acceptable for production if operational burden is low.")
        else:
            print("    [STRONG] XGBoost earns its complexity.")
            print("             Significant improvement justifies the operational overhead.")

    # ── XGBoost vs Linear ────────────────────────────────────────────────────
    if linear and xgboost:
        linear_improvement = (
            (linear["rmse_seconds"] - xgboost["rmse_seconds"])
            / linear["rmse_seconds"] * 100
        )
        print(f"\n  XGBoost vs Linear Regression:")
        print(f"    RMSE improvement: {linear_improvement:.1f}%")

        if linear_improvement < 5:
            print("    [SIGNAL] XGBoost barely beats Linear Regression.")
            print("             Your data may be largely linear.")
            print("             Linear Regression is simpler, faster, more interpretable.")
            print("             It may be the better production choice.")
        elif linear_improvement < 20:
            print("    [SIGNAL] Moderate non-linearity in the data.")
            print("             XGBoost capturing some interactions Linear cannot.")
        else:
            print("    [SIGNAL] Strong non-linearity confirmed.")
            print("             XGBoost is capturing complex interactions.")
            print("             'Rain + Friday + Midtown' type patterns are real.")

    # ── Bias analysis ────────────────────────────────────────────────────────
    print(f"\n  Bias Analysis (systematic prediction error):")
    for name, m in results.items():
        bias = m["mean_error_seconds"]
        if abs(bias) < 10:
            verdict = "well-calibrated"
        elif bias > 0:
            verdict = f"over-predicts by {bias:.0f}s on average — customers get pleasant surprises"
        else:
            verdict = f"under-predicts by {abs(bias):.0f}s on average — customers arrive late"
        print(f"    {name:<24}: {verdict}")

    # ── Tail failure analysis ────────────────────────────────────────────────
    print(f"\n  Tail Failure Analysis (P95 — worst 5% of predictions):")
    for name, m in results.items():
        p95 = m["p95_error_seconds"]
        print(f"    {name:<24}: worst 5% of predictions off by {p95:.0f}s ({p95/60:.1f} min)")

    if xgboost:
        p95 = xgboost["p95_error_seconds"]
        print(f"\n    In production, that {p95/60:.1f} min tail error lands on")
        print(f"    your highest-demand anomaly scenarios — concerts, storms, strikes.")
        print(f"    Standard RMSE monitoring will miss this. P95 monitoring catches it.")

    # ── Overfit warning ──────────────────────────────────────────────────────
    if xgboost and "overfit_gap_pct" in xgboost:
        gap = xgboost["overfit_gap_pct"]
        print(f"\n  XGBoost Overfit Gap: {gap:.1f}%")
        if gap > 30:
            print("    [WARNING] Gap > 30%. Model may not generalise to production traffic.")
            print("              Increase reg_lambda or min_child_weight before deploying.")
        elif gap > 15:
            print("    [WATCH]   Gap 15-30%. Mild overfit. Monitor production closely.")
        else:
            print("    [OK]      Gap < 15%. Model generalises well.")

    print("\n" + "═" * 95)
    print("  Silent Failure Reminder:")
    print("  These metrics were computed on historical validation data.")
    print("  They say nothing about model correctness on future anomalous events.")
    print("  A concert, strike, or storm will degrade all three models equally.")
    print("  The defense is production monitoring — not better validation metrics.")
    print("  See: docs/silent_failure.md")
    print("═" * 95 + "\n")


def main():
    results = load_metrics()
    print_comparison_table(results)
    print_verdicts(results)


if __name__ == "__main__":
    main()
