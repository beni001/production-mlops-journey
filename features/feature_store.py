"""
features/feature_store.py — The Feature Store Interface.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features.features import FEATURES, TARGET, haversine
from features.feature_views import compute_point_in_time_features

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "registry.json")

def load_registry():
    with open(REGISTRY_PATH) as f:
        return json.load(f)

def list_features():
    registry = load_registry()
    print(f"\nFEATURE REGISTRY v{registry['version']}")
    for view in registry["feature_views"]:
        print(f"\n[{view['name']}]")
        for feat in view["features"]:
            print(f"  {feat['name']}")

def validate_feature_request(feature_list):
    registry = load_registry()
    all_features = {
        feat["name"]
        for view in registry["feature_views"]
        for feat in view["features"]
    }
    unknown = set(feature_list) - all_features
    if unknown:
        raise ValueError(f"REGISTRY ERROR: Unknown features: {unknown}")

def get_historical_features(data_path, feature_list=None, include_point_in_time=True, validate_pit=False):
    print("\n[feature_store] Loading historical features...")
    if feature_list:
        validate_feature_request(feature_list)
    else:
        feature_list = FEATURES

    df = pd.read_parquet(data_path)
    print(f"[feature_store] Loaded: {len(df):,} rows")

    if include_point_in_time:
        df = compute_point_in_time_features(df)
        pit_features = ["trip_count_last_1h", "avg_duration_same_hour_last7days"]
        for f in pit_features:
            if f not in feature_list:
                feature_list = feature_list + [f]

    df = df.sort_values("pickup_datetime").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.75)
    gap_end   = int(n * 0.80)
    train = df.iloc[:train_end].copy()
    val   = df.iloc[gap_end:].copy()

    print(f"[feature_store] Train: {len(train):,} | Val: {len(val):,}")

    available = [f for f in feature_list if f in df.columns]
    missing   = [f for f in feature_list if f not in df.columns]
    if missing:
        print(f"[feature_store] WARNING: not in data: {missing}")

    return train, val, available

def get_online_features(request):
    print("[feature_store] Computing online features...")
    if isinstance(request["pickup_datetime"], str):
        pickup_dt = pd.to_datetime(request["pickup_datetime"])
    else:
        pickup_dt = request["pickup_datetime"]

    hour         = pickup_dt.hour
    day_of_week  = pickup_dt.dayofweek
    month        = pickup_dt.month
    is_weekend   = int(day_of_week >= 5)
    is_rush_hour = int(hour in [7, 8, 9, 17, 18, 19])
    distance_km  = haversine(
        request["pickup_latitude"],  request["pickup_longitude"],
        request["dropoff_latitude"], request["dropoff_longitude"]
    )

    features = {
        "hour":              hour,
        "day_of_week":       day_of_week,
        "month":             month,
        "is_weekend":        is_weekend,
        "is_rush_hour":      is_rush_hour,
        "distance_km":       float(distance_km),
        "passenger_count":   int(request["passenger_count"]),
        "pickup_latitude":   float(request["pickup_latitude"]),
        "pickup_longitude":  float(request["pickup_longitude"]),
        "dropoff_latitude":  float(request["dropoff_latitude"]),
        "dropoff_longitude": float(request["dropoff_longitude"]),
        "vendor_id":         int(request["vendor_id"]),
    }
    for k, v in features.items():
        print(f"  {k:<28}: {v}")
    return features
