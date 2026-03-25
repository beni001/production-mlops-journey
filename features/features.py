"""
features/features.py — Single source of truth for feature engineering.

Architectural contract (see docs/architecture.md):
  "Feature definitions are centralized. Shared between training and inference."

Both Linear Regression and XGBoost import from here.
The inference server (Day 6) imports from here.
The ETL pipeline (Day 7) imports from here.

If you change a feature here, it changes everywhere simultaneously.
That is the point. Training-serving skew is impossible if there is
only one place where features are defined.
"""

import pandas as pd
import numpy as np

# ── The canonical feature list ─────────────────────────────────────────────
# This is the contract. The model artifact on disk was trained on exactly
# these features in exactly this order. Inference must present them in the
# same order or predictions are silently wrong (column swap = silent failure).
FEATURES = [
    "hour",               # 0-23 — captures time-of-day demand patterns
    "day_of_week",        # 0=Monday, 6=Sunday — captures weekly patterns
    "month",              # 1-12 — captures seasonal patterns
    "is_weekend",         # binary — weekends have fundamentally different demand
    "is_rush_hour",       # binary — 7-9am and 5-7pm are structurally different
    "distance_km",        # haversine distance — strongest predictor of duration
    "passenger_count",    # weak predictor but available at request time
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "vendor_id",          # 1 or 2 — vendors have different route preferences
]

TARGET = "trip_duration"  # seconds — regression target


def haversine(lat1, lon1, lat2, lon2):
    """
    Straight-line distance between two GPS coordinates in kilometres.

    Why haversine and not Euclidean distance:
    Euclidean distance treats the Earth as flat. At NYC's latitude
    (~40°N), one degree of longitude ≠ one degree of latitude in km.
    Euclidean would systematically underestimate east-west distances.
    Haversine accounts for Earth's curvature — correct for any city.

    This runs on numpy arrays — no Python loops, vectorized over
    the entire dataframe in one operation.
    """
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def load_and_engineer(path: str) -> pd.DataFrame:
    """
    Load raw CSV and engineer all features.

    Data quality filters applied here:
    - trip_duration < 60s    : false starts, GPS glitches, data entry errors
    - trip_duration > 7200s  : 2 hour cap — outliers that destroy loss functions
    - distance_km < 0.1      : GPS coordinate failures (pickup = dropoff)
    - passenger_count < 1    : invalid records

    These filters are deterministic and documented.
    If the raw data changes, the filters stay the same.
    If you need different filters, change them here — they apply
    everywhere automatically.
    """
    print(f"[features] Loading {path}")
    df = pd.read_csv(path, parse_dates=["pickup_datetime", "dropoff_datetime"])
    print(f"[features] Raw rows: {len(df):,}")

    # ── Temporal features ───────────────────────────────────────────────────
    df["hour"]         = df["pickup_datetime"].dt.hour
    df["day_of_week"]  = df["pickup_datetime"].dt.dayofweek
    df["month"]        = df["pickup_datetime"].dt.month
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    # ── Spatial features ────────────────────────────────────────────────────
    df["distance_km"] = haversine(
        df["pickup_latitude"],  df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"]
    )

    # ── Data quality filters ────────────────────────────────────────────────
    before = len(df)
    df = df[(df[TARGET] > 60) & (df[TARGET] < 7200)]
    df = df[df["distance_km"] > 0.1]
    df = df[df["passenger_count"] > 0]
    df = df.dropna(subset=FEATURES + [TARGET])
    after = len(df)
    print(f"[features] Removed {before - after:,} corrupt rows ({(before-after)/before*100:.1f}%)")
    print(f"[features] Clean rows: {after:,}")

    return df


def temporal_split(df: pd.DataFrame, train_frac=0.75, gap_frac=0.05):
    """
    Split data by time — never randomly.

    Why temporal split:
    Rideshare data is time-correlated. A random split allows the model
    to see trips from next week while training on this week's data.
    That is data leakage — the model learns future patterns it would
    never have in production. Validation metrics become optimistic lies.

    Why the gap period:
    Trips at the boundary of train and validation share temporal
    autocorrelation — Friday patterns bleed into Saturday.
    The gap period (default 5%) is discarded entirely. Its only job
    is to break the autocorrelation at the boundary.
    Cost: 5% of data. Benefit: honest validation metrics.

    Split: 75% train | 5% gap (discarded) | 20% validate
    """
    df = df.sort_values("pickup_datetime").reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_frac)
    gap_end   = int(n * (train_frac + gap_frac))

    train = df.iloc[:train_end].copy()
    # gap  = df.iloc[train_end:gap_end]  ← discarded, never used
    val   = df.iloc[gap_end:].copy()

    print(f"\n[features] ── Temporal Split ──────────────────────")
    print(f"[features] Train : {len(train):,} rows")
    print(f"[features]   from: {train['pickup_datetime'].min()}")
    print(f"[features]   to  : {train['pickup_datetime'].max()}")
    print(f"[features] Gap   : {gap_end - train_end:,} rows discarded")
    print(f"[features] Val   : {len(val):,} rows")
    print(f"[features]   from: {val['pickup_datetime'].min()}")
    print(f"[features]   to  : {val['pickup_datetime'].max()}")
    print(f"[features] ────────────────────────────────────────\n")

    # Leakage check — val must start strictly after train ends
    assert val["pickup_datetime"].min() > train["pickup_datetime"].max(), \
        "LEAKAGE DETECTED: validation data overlaps with training data"

    return train, val


def compute_metrics(actual: pd.Series, predicted: np.ndarray, model_name: str) -> dict:
    """
    Five-metric evaluation contract.

    Why five metrics and not just RMSE:

    rmse_seconds      : overall error magnitude. Penalises large errors heavily.
                        The primary comparison metric.

    mape_pct          : percentage error. Scale-invariant.
                        "Off by 10%" means the same whether trip is 5min or 50min.

    mean_error_seconds: SIGNED average error. Reveals systematic bias.
                        A model with RMSE=180s might be consistently under-predicting
                        by 3 minutes on every single trip. RMSE hides this.
                        Positive = over-predicting. Negative = under-predicting.

    p95_error_seconds : 95th percentile absolute error. Reveals tail failure.
                        A model can have excellent average performance while being
                        catastrophically wrong on 5% of trips.
                        In production, that 5% is exactly the concert/storm/strike
                        scenarios — the highest-stakes predictions.

    r2                : variance explained. 1.0 = perfect. 0.0 = predicts the mean.
                        Negative = worse than predicting the mean every time.
    """
    errors    = actual - predicted
    abs_errors = np.abs(errors)

    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(abs_errors / actual) * 100
    me   = np.mean(errors)                          # signed
    p95  = np.percentile(abs_errors, 95)
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2   = 1 - (ss_res / ss_tot)

    return {
        "model":               model_name,
        "rmse_seconds":        round(float(rmse), 2),
        "mape_pct":            round(float(mape), 2),
        "mean_error_seconds":  round(float(me),   2),
        "p95_error_seconds":   round(float(p95),  2),
        "r2":                  round(float(r2),   4),
    }
