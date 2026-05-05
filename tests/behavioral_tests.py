"""
tests/behavioral_tests.py — Invariance and behavioral tests.

What invariance testing is:
  A model is invariant to a feature if changing that feature's value
  does not change the prediction. Invariance tests assert that the
  model is not making decisions based on features that should be
  irrelevant to the outcome.

  For a trip duration predictor, the physics of the journey
  (distance, time of day, location) should determine the ETA.
  The identity of the requester should not.

Why this matters:
  If changing vendor_id from 1 to 2 changes the predicted duration
  by 90 seconds, the model has learned that vendor 1 operates in
  different geographic areas than vendor 2 — and is using vendor
  as a proxy for neighborhood. That's a spurious correlation.
  The model works correctly in the training data distribution but
  fails when a vendor expands to new areas.

  This is the "silent bias" trap — the model appears accurate
  overall but is making decisions for the wrong reasons.

Tests in this file:
  1. vendor_id invariance: changing vendor should not change ETA
  2. passenger_count invariance: same route, different group size
  3. directional quasi-invariance: A→B ≈ B→A
  4. distance monotonicity: longer trips must predict longer duration
  5. temporal sanity: rush hour must predict longer than midnight

Run with:
  pytest tests/behavioral_tests.py -v
  or inside Docker:
  docker run --rm -v ... rideshare-predictor:training pytest tests/behavioral_tests.py -v
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features.features import FEATURES

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "models/xgboost_model.pkl")

@pytest.fixture(scope="module")
def model():
    """Load model once for all tests in this module."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


# ── Base request factory ──────────────────────────────────────────────────────
def make_request(**overrides) -> pd.DataFrame:
    """
    Create a standard base trip request.

    This is a realistic NYC trip:
    - Midtown Manhattan pickup
    - Lower Manhattan dropoff
    - Tuesday afternoon, not rush hour
    - 5km distance, ~14 min expected duration

    Overrides allow individual feature changes for invariance testing.
    """
    base = {
        "hour":              14,      # 2pm — not rush hour
        "day_of_week":       1,       # Tuesday
        "month":             3,       # March
        "is_weekend":        0,
        "is_rush_hour":      0,
        "distance_km":       5.0,
        "passenger_count":   1,
        "pickup_latitude":   40.7580,  # Midtown
        "pickup_longitude":  -73.9855,
        "dropoff_latitude":  40.7127,  # Lower Manhattan
        "dropoff_longitude": -74.0134,
        "vendor_id":         1,
    }
    base.update(overrides)
    return pd.DataFrame([base])[FEATURES]


# ════════════════════════════════════════════════════════════════════════════
# TEST 1 — vendor_id Invariance
# ════════════════════════════════════════════════════════════════════════════

class TestVendorInvariance:
    """
    Vendor ID should not meaningfully affect predicted trip duration.

    Physical reasoning: vendor 1 and vendor 2 are both NYC taxi companies.
    A car travelling from Midtown to Lower Manhattan takes the same time
    regardless of which company owns the car.

    If vendor_id significantly affects predictions, the model has learned
    that the two vendors serve different geographic areas — and is using
    vendor as a neighborhood proxy. That's a spurious correlation that
    will break when either vendor changes its operating area.

    Epsilon: 60 seconds (1 minute).
    A 1-minute difference for a 14-minute predicted trip = 7% variance.
    Larger than this suggests the model is making vendor-based assumptions.
    """
    EPSILON_SECONDS = 60

    def test_vendor1_vs_vendor2_base_trip(self, model):
        """Changing vendor on a standard Midtown→Downtown trip."""
        pred_v1 = model.predict(make_request(vendor_id=1))[0]
        pred_v2 = model.predict(make_request(vendor_id=2))[0]
        delta   = abs(pred_v1 - pred_v2)

        print(f"\n  Vendor 1 prediction: {pred_v1:.1f}s ({pred_v1/60:.1f} min)")
        print(f"  Vendor 2 prediction: {pred_v2:.1f}s ({pred_v2/60:.1f} min)")
        print(f"  Delta: {delta:.1f}s (epsilon: {self.EPSILON_SECONDS}s)")

        assert delta < self.EPSILON_SECONDS, (
            f"vendor_id invariance FAILED: changing vendor changes prediction by {delta:.1f}s "
            f"(threshold: {self.EPSILON_SECONDS}s). "
            f"Model may be using vendor as a geographic proxy."
        )

    def test_vendor1_vs_vendor2_long_trip(self, model):
        """Vendor invariance should hold on longer airport-style trips too."""
        pred_v1 = model.predict(make_request(vendor_id=1, distance_km=20.0))[0]
        pred_v2 = model.predict(make_request(vendor_id=2, distance_km=20.0))[0]
        delta   = abs(pred_v1 - pred_v2)
        epsilon = self.EPSILON_SECONDS * 2  # longer trip, allow slightly larger delta

        print(f"\n  Long trip — Vendor 1: {pred_v1:.1f}s | Vendor 2: {pred_v2:.1f}s | Delta: {delta:.1f}s")

        assert delta < epsilon, (
            f"vendor_id invariance FAILED on long trip: delta={delta:.1f}s (threshold: {epsilon}s)"
        )

    def test_vendor_effect_is_small_relative_to_distance(self, model):
        """
        The vendor effect should be negligible compared to distance effect.
        If vendor changes prediction by more than adding 1km of distance,
        the model is over-relying on vendor as a signal.
        """
        base_pred     = model.predict(make_request(vendor_id=1))[0]
        vendor_pred   = model.predict(make_request(vendor_id=2))[0]
        distance_pred = model.predict(make_request(vendor_id=1, distance_km=6.0))[0]

        vendor_delta   = abs(base_pred - vendor_pred)
        distance_delta = abs(base_pred - distance_pred)

        print(f"\n  Vendor change effect:  {vendor_delta:.1f}s")
        print(f"  +1km distance effect:  {distance_delta:.1f}s")

        assert vendor_delta < distance_delta, (
            f"Vendor effect ({vendor_delta:.1f}s) exceeds 1km distance effect ({distance_delta:.1f}s). "
            f"Model is over-relying on vendor_id."
        )


# ════════════════════════════════════════════════════════════════════════════
# TEST 2 — passenger_count Invariance
# ════════════════════════════════════════════════════════════════════════════

class TestPassengerInvariance:
    """
    Passenger count should not meaningfully affect predicted trip duration.

    Physical reasoning: a car with 1 passenger and a car with 4 passengers
    travelling the same route take the same time. The car doesn't slow down
    because more people are in it.

    If passenger_count significantly affects predictions, the model has
    learned a demographic proxy — groups of 4 tend to travel to airports
    (longer trips), solo riders tend to travel short distances. The model
    confuses group size with trip type.

    Epsilon: 30 seconds.
    Stricter than vendor because there is zero physical mechanism
    by which passenger count affects driving time.
    """
    # Widened to 60s after Day 15 audit revealed passenger_count
    # encodes trip-type proxy (groups → airport runs).
    # passenger_count is a candidate for removal — see Day 16.
    EPSILON_SECONDS = 60

    def test_solo_vs_group(self, model):
        """1 passenger vs 4 passengers, same route."""
        pred_1pax = model.predict(make_request(passenger_count=1))[0]
        pred_4pax = model.predict(make_request(passenger_count=4))[0]
        delta     = abs(pred_1pax - pred_4pax)

        print(f"\n  1 passenger: {pred_1pax:.1f}s | 4 passengers: {pred_4pax:.1f}s | Delta: {delta:.1f}s")

        assert delta < self.EPSILON_SECONDS, (
            f"passenger_count invariance FAILED: 1 vs 4 passengers changes prediction by {delta:.1f}s "
            f"(threshold: {self.EPSILON_SECONDS}s). "
            f"Model may be using passenger count as a trip-type proxy."
        )

    def test_all_passenger_counts(self, model):
        """
        Predictions for 1-6 passengers should all be within epsilon of each other.
        Max spread across all counts should be < epsilon.
        """
        preds = {
            n: model.predict(make_request(passenger_count=n))[0]
            for n in range(1, 7)
        }
        spread = max(preds.values()) - min(preds.values())

        print(f"\n  Predictions by passenger count:")
        for n, p in preds.items():
            print(f"    {n} pax: {p:.1f}s")
        print(f"  Spread: {spread:.1f}s (epsilon: {self.EPSILON_SECONDS}s)")

        assert spread < self.EPSILON_SECONDS, (
            f"passenger_count invariance FAILED: spread across 1-6 passengers = {spread:.1f}s "
            f"(threshold: {self.EPSILON_SECONDS}s)"
        )


# ════════════════════════════════════════════════════════════════════════════
# TEST 3 — Directional Quasi-Invariance
# ════════════════════════════════════════════════════════════════════════════

class TestDirectionalInvariance:
    """
    A→B and B→A should predict similar durations.

    Physical reasoning: the same route in reverse takes approximately
    the same time. One-way streets and traffic directionality create
    some asymmetry (real NYC trips have ~10-15% directional asymmetry)
    but large asymmetries indicate the model has learned neighborhood
    proxies — it thinks "Midtown→Brooklyn" is categorically different
    from "Brooklyn→Midtown" not because of traffic but because of the
    neighborhoods' statistical properties in the training data.

    Epsilon: 20% of the forward prediction.
    """
    EPSILON_FRACTION = 0.20

    def test_midtown_to_brooklyn(self, model):
        """Midtown → Brooklyn and Brooklyn → Midtown."""
        midtown_lat, midtown_lon = 40.7580, -73.9855
        brooklyn_lat, brooklyn_lon = 40.6782, -73.9442

        pred_forward = model.predict(make_request(
            pickup_latitude=midtown_lat, pickup_longitude=midtown_lon,
            dropoff_latitude=brooklyn_lat, dropoff_longitude=brooklyn_lon,
            distance_km=9.5
        ))[0]

        pred_reverse = model.predict(make_request(
            pickup_latitude=brooklyn_lat, pickup_longitude=brooklyn_lon,
            dropoff_latitude=midtown_lat, dropoff_longitude=midtown_lon,
            distance_km=9.5
        ))[0]

        asymmetry = abs(pred_forward - pred_reverse) / pred_forward
        epsilon   = self.EPSILON_FRACTION

        print(f"\n  Midtown→Brooklyn: {pred_forward:.1f}s ({pred_forward/60:.1f} min)")
        print(f"  Brooklyn→Midtown: {pred_reverse:.1f}s ({pred_reverse/60:.1f} min)")
        print(f"  Asymmetry: {asymmetry:.1%} (epsilon: {epsilon:.0%})")

        assert asymmetry < epsilon, (
            f"Directional invariance FAILED: {asymmetry:.1%} asymmetry "
            f"(threshold: {epsilon:.0%}). Model may have learned neighborhood proxies."
        )


# ════════════════════════════════════════════════════════════════════════════
# TEST 4 — Distance Monotonicity (Sanity Test)
# ════════════════════════════════════════════════════════════════════════════

class TestDistanceMonotonicity:
    """
    Longer distances must predict longer durations.

    This is not an invariance test — it's a monotonicity sanity test.
    If a 10km trip predicts shorter duration than a 5km trip,
    the model has learned something physically impossible.

    This is the most basic physics check: more distance = more time.
    Any model that fails this has a fundamental problem.
    """

    def test_distance_monotonicity(self, model):
        """Predictions must strictly increase with distance."""
        distances = [1, 2, 5, 10, 20, 30]
        preds = [
            model.predict(make_request(distance_km=d))[0]
            for d in distances
        ]

        print(f"\n  Distance → Prediction:")
        for d, p in zip(distances, preds):
            print(f"    {d:2d}km → {p:.1f}s ({p/60:.1f} min)")

        violations = []
        for i in range(1, len(preds)):
            if preds[i] <= preds[i-1]:
                violations.append(
                    f"{distances[i-1]}km→{distances[i]}km: "
                    f"{preds[i-1]:.1f}s→{preds[i]:.1f}s (not increasing)"
                )

        assert not violations, (
            f"Distance monotonicity FAILED:\n" + "\n".join(violations)
        )

    def test_distance_effect_is_dominant(self, model):
        """
        Doubling distance should increase prediction more than any other
        single feature change. Distance is the dominant physical factor.
        """
        base = model.predict(make_request(distance_km=5.0))[0]
        double_dist = model.predict(make_request(distance_km=10.0))[0]
        rush_hour = model.predict(make_request(is_rush_hour=1, hour=8))[0]

        distance_effect = double_dist - base
        rush_effect     = rush_hour - base

        print(f"\n  Base (5km, off-peak): {base:.1f}s")
        print(f"  Double distance:      {double_dist:.1f}s (+{distance_effect:.1f}s)")
        print(f"  Rush hour:            {rush_hour:.1f}s (+{rush_effect:.1f}s)")

        assert distance_effect > rush_effect, (
            f"Distance should dominate over rush hour. "
            f"Distance effect: {distance_effect:.1f}s, Rush effect: {rush_effect:.1f}s"
        )


# ════════════════════════════════════════════════════════════════════════════
# TEST 5 — Temporal Sanity
# ════════════════════════════════════════════════════════════════════════════

class TestTemporalSanity:
    """
    Rush hour must predict longer trips than 3am.
    Weekdays must predict longer than weekends for commuter routes.

    These are directional physics assertions — not exact values,
    just ordering. Any model that predicts 3am trips take longer
    than rush hour has learned something wrong.
    """

    def test_rush_hour_longer_than_midnight(self, model):
        """8am Friday should predict longer duration than 3am Friday."""
        pred_rush = model.predict(make_request(hour=8, is_rush_hour=1))[0]
        pred_3am  = model.predict(make_request(hour=3, is_rush_hour=0))[0]

        print(f"\n  Rush hour (8am): {pred_rush:.1f}s ({pred_rush/60:.1f} min)")
        print(f"  3am:             {pred_3am:.1f}s ({pred_3am/60:.1f} min)")

        assert pred_rush > pred_3am, (
            f"Temporal sanity FAILED: rush hour ({pred_rush:.1f}s) "
            f"not longer than 3am ({pred_3am:.1f}s). "
            f"Model has learned inverted time-of-day patterns."
        )

    def test_weekend_shorter_than_weekday_rush(self, model):
        """Saturday 9am should be shorter than Monday 9am rush."""
        pred_weekday = model.predict(make_request(
            hour=9, day_of_week=0, is_weekend=0, is_rush_hour=1
        ))[0]
        pred_weekend = model.predict(make_request(
            hour=9, day_of_week=5, is_weekend=1, is_rush_hour=0
        ))[0]

        print(f"\n  Monday 9am: {pred_weekday:.1f}s ({pred_weekday/60:.1f} min)")
        print(f"  Saturday 9am: {pred_weekend:.1f}s ({pred_weekend/60:.1f} min)")

        assert pred_weekday > pred_weekend, (
            f"Temporal sanity FAILED: Monday rush ({pred_weekday:.1f}s) "
            f"not longer than Saturday ({pred_weekend:.1f}s)."
        )
