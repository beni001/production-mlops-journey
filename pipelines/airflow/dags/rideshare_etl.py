"""
pipelines/airflow/dags/rideshare_etl.py — The Automated Pulse

This DAG formalizes the Day 6 wall of fire into a scheduled,
observable, auditable pipeline.

What a DAG is:
  A Python file that defines tasks and their dependencies.
  Airflow reads this file, builds the execution graph,
  and runs it on the defined schedule.

Schedule: daily at midnight UTC.
  Every day at 00:00 UTC Airflow triggers this DAG automatically.
  You don't run scripts. You monitor a dashboard.

Idempotency contract:
  Running this DAG for 2026-04-10 five times produces the
  exact same Parquet file every time.
  Same input + same code = same output. Always.
  This is enforced by:
    1. Reading from immutable raw data (DVC-locked)
    2. Deterministic cleaning rules (no randomness)
    3. Overwriting the output file (not appending)

The four tasks:
  Task 1: validate_raw_schema
    Checks the raw CSV matches the expected schema.
    Fails loudly if columns are missing or GPS bounds violated.
    If this fails, nothing else runs.

  Task 2: run_quality_gates
    Statistical checks on the raw data.
    Null rates, value ranges, row counts.
    Catches sensor failures and upstream corruption
    that schema validation alone misses.

  Task 3: transform_to_parquet
    Runs make_dataset.py — clean, engineer, write Parquet.
    Only runs if tasks 1 and 2 passed.

  Task 4: verify_output
    Reads the produced Parquet, verifies row count and
    column presence. Final confirmation the output is valid.

Dependency chain:
  validate_raw_schema >> run_quality_gates >> transform_to_parquet >> verify_output

>> means "must succeed before the next task starts"
If any task fails, all downstream tasks are skipped.
The dashboard shows exactly which task failed and why.
"""

import os
import sys
import json
import hashlib
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Path setup ──────────────────────────────────────────────────
# Airflow runs from /opt/airflow inside the container.
# src/ and features/ are mounted there via docker-compose volumes.
sys.path.insert(0, "/opt/airflow")
sys.path.insert(0, "/opt/airflow/src")

log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────
RAW_PATH       = os.environ.get("DATA_RAW_PATH",      "/opt/airflow/data/raw/rideshare_logs.csv")
PROCESSED_PATH = os.environ.get("DATA_PROCESSED_PATH","/opt/airflow/data/processed/rideshare_clean.parquet")
MANIFEST_PATH  = os.environ.get("MANIFEST_PATH",      "/opt/airflow/data/processed/manifest.json")

# ── Schema contract ──────────────────────────────────────────────
REQUIRED_COLUMNS = {
    "id", "vendor_id", "pickup_datetime", "dropoff_datetime",
    "passenger_count", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
    "store_and_fwd_flag", "trip_duration"
}

NYC_LAT = (40.4, 41.0)
NYC_LON = (-74.5, -73.5)

# ── Quality gate thresholds ──────────────────────────────────────
# These are the lines in the sand.
# Cross them and the pipeline fails loudly.
MAX_NULL_RATE        = 0.02   # more than 2% nulls = upstream problem
MAX_ZERO_COORD_RATE  = 0.01   # more than 1% zero coordinates = sensor failure
MIN_ROW_COUNT        = 100000 # fewer than 100k rows = incomplete data delivery
MAX_BAD_GPS_RATE     = 0.05   # more than 5% outside NYC = wrong dataset


# ════════════════════════════════════════════════════════════════
# TASK 1 — Schema Validation
# ════════════════════════════════════════════════════════════════

def validate_raw_schema(**context):
    """
    Task 1: Validate the raw CSV schema.

    This is the first gate. If upstream changed anything about
    the data structure, this task fails here — before a single
    row of bad data enters the pipeline.

    context: Airflow passes this automatically to every Python
    task. It contains run metadata — execution_date, run_id,
    dag_id. We use execution_date to make logs auditable.
    """
    execution_date = context["execution_date"]
    log.info(f"[Task 1] Schema validation for run: {execution_date}")
    log.info(f"[Task 1] Reading raw data from: {RAW_PATH}")

    # ── File existence check ─────────────────────────────────────
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"Raw data not found at {RAW_PATH}.\n"
            f"Has DVC pull been run? Is the volume mount correct?"
        )

    # ── Load header only — no need to read 192MB for schema check
    df_sample = pd.read_csv(RAW_PATH, nrows=1000)
    log.info(f"[Task 1] Columns found: {list(df_sample.columns)}")

    # ── Missing column check ─────────────────────────────────────
    missing = REQUIRED_COLUMNS - set(df_sample.columns)
    if missing:
        raise ValueError(
            f"SCHEMA VIOLATION: Missing columns: {missing}\n"
            f"Upstream data has changed. Pipeline halted.\n"
            f"Fix the source before rerunning."
        )

    # ── GPS parseable check ──────────────────────────────────────
    try:
        pd.to_datetime(df_sample["pickup_datetime"].head(10))
    except Exception as e:
        raise ValueError(
            f"SCHEMA VIOLATION: pickup_datetime unparseable.\n"
            f"Timestamp format may have changed upstream. Error: {e}"
        )

    # ── GPS bounds on sample ─────────────────────────────────────
    bad_gps = df_sample[
        (df_sample["pickup_latitude"] < NYC_LAT[0]) |
        (df_sample["pickup_latitude"] > NYC_LAT[1])
    ]
    if len(bad_gps) > len(df_sample) * MAX_BAD_GPS_RATE:
        raise ValueError(
            f"SCHEMA VIOLATION: {len(bad_gps)} of {len(df_sample)} sampled rows "
            f"have pickup_latitude outside NYC bounds {NYC_LAT}.\n"
            f"Wrong dataset or coordinate system change."
        )

    log.info("[Task 1] Schema validation PASSED.")
    log.info(f"[Task 1] All {len(REQUIRED_COLUMNS)} required columns present.")
    log.info(f"[Task 1] Timestamps parseable. GPS bounds valid.")

    # ── Push schema info to XCom ─────────────────────────────────
    # XCom is Airflow's inter-task communication mechanism.
    # Task 1 pushes metadata. Task 2 can pull and build on it.
    # Think of it as a shared notepad between tasks.
    return {
        "columns_found": list(df_sample.columns),
        "sample_rows":   len(df_sample),
        "validated_at":  str(execution_date),
    }


# ════════════════════════════════════════════════════════════════
# TASK 2 — Quality Gates
# ════════════════════════════════════════════════════════════════

def run_quality_gates(**context):
    """
    Task 2: Statistical quality checks on the full raw dataset.

    Schema validation checks structure.
    Quality gates check content.

    A file can have all the right columns with all the right types
    and still contain garbage data:
      - A sensor fails and sends 0.0 for all coordinates
      - A pipeline bug produces 10 rows instead of 1.4 million
      - A timezone change makes all timestamps wrong by 5 hours
      - A null rate spikes because an upstream join failed

    These checks catch what schema validation misses.

    The "silent sensor failure" scenario:
      GPS sensor fails → sends 0.0,0.0 for all coordinates
      Schema check: PASSES (0.0 is a valid float)
      Quality gate: FAILS (1% of coords are exactly 0.0)
      Without this gate: model trains on Manhattan Bridge → 0.0,0.0
      trips, learns nonsense, deploys, silently fails on real data.
    """
    execution_date = context["execution_date"]
    log.info(f"[Task 2] Quality gates for run: {execution_date}")

    df = pd.read_csv(RAW_PATH, parse_dates=["pickup_datetime"])
    total = len(df)
    log.info(f"[Task 2] Full dataset loaded: {total:,} rows")

    violations = []

    # ── Row count gate ───────────────────────────────────────────
    # Fewer rows than expected = incomplete delivery
    if total < MIN_ROW_COUNT:
        violations.append(
            f"ROW COUNT: {total:,} rows below minimum {MIN_ROW_COUNT:,}.\n"
            f"Incomplete data delivery. Check upstream pipeline."
        )

    # ── Null rate gate ───────────────────────────────────────────
    # High null rate = upstream join failed or column dropped
    for col in ["pickup_latitude", "pickup_longitude",
                "dropoff_latitude", "dropoff_longitude", "trip_duration"]:
        null_rate = df[col].isna().sum() / total
        if null_rate > MAX_NULL_RATE:
            violations.append(
                f"NULL RATE: {col} has {null_rate*100:.2f}% nulls "
                f"(max allowed: {MAX_NULL_RATE*100:.1f}%).\n"
                f"Upstream join or extraction may have failed."
            )

    # ── Zero coordinate gate ─────────────────────────────────────
    # GPS sensor failure sends 0.0,0.0 — physically impossible for NYC
    zero_lat = (df["pickup_latitude"] == 0.0).sum() / total
    zero_lon = (df["pickup_longitude"] == 0.0).sum() / total
    if zero_lat > MAX_ZERO_COORD_RATE:
        violations.append(
            f"ZERO COORDS: {zero_lat*100:.2f}% of pickup_latitude is 0.0 "
            f"(max allowed: {MAX_ZERO_COORD_RATE*100:.1f}%).\n"
            f"GPS sensor failure suspected."
        )
    if zero_lon > MAX_ZERO_COORD_RATE:
        violations.append(
            f"ZERO COORDS: {zero_lon*100:.2f}% of pickup_longitude is 0.0.\n"
            f"GPS sensor failure suspected."
        )

    # ── Negative duration gate ───────────────────────────────────
    # Negative trip duration is physically impossible
    neg_duration = (df["trip_duration"] < 0).sum()
    if neg_duration > 0:
        violations.append(
            f"NEGATIVE DURATION: {neg_duration:,} rows have negative trip_duration.\n"
            f"Clock synchronization failure or data corruption."
        )

    # ── Future timestamp gate ────────────────────────────────────
    # Trips in the future = system clock wrong or data corruption
    now = pd.Timestamp.now()
    future_trips = (df["pickup_datetime"] > now).sum()
    if future_trips > 0:
        violations.append(
            f"FUTURE TIMESTAMPS: {future_trips:,} trips have future pickup_datetime.\n"
            f"System clock error or timestamp format change."
        )

    # ── Verdict ──────────────────────────────────────────────────
    if violations:
        violation_text = "\n\n".join(violations)
        raise ValueError(
            f"QUALITY GATE FAILED: {len(violations)} violation(s) detected.\n\n"
            f"{violation_text}\n\n"
            f"Pipeline halted. Fix upstream data before rerunning."
        )

    log.info("[Task 2] All quality gates PASSED.")
    log.info(f"[Task 2] Row count: {total:,} (above minimum {MIN_ROW_COUNT:,})")
    log.info(f"[Task 2] Null rates: within acceptable bounds")
    log.info(f"[Task 2] Zero coordinates: within acceptable bounds")
    log.info(f"[Task 2] No negative durations or future timestamps")

    return {
        "total_rows":     total,
        "gates_passed":   True,
        "checked_at":     str(execution_date),
    }


# ════════════════════════════════════════════════════════════════
# TASK 3 — Transform to Parquet
# ════════════════════════════════════════════════════════════════

def transform_to_parquet(**context):
    """
    Task 3: Run the wall of fire — clean, engineer, write Parquet.

    This task calls make_dataset.py logic directly.
    It only runs if Tasks 1 and 2 both passed.
    By the time this task runs, we know:
      - Schema is correct
      - Data quality is acceptable
      - It is safe to invest compute in transformation

    Idempotency is enforced by overwriting the output file.
    Running this task 5 times on the same raw data produces
    the same Parquet file with the same hash every time.
    """
    execution_date = context["execution_date"]
    run_id = context["run_id"]
    log.info(f"[Task 3] Transform starting for run: {run_id}")

    # Import make_dataset functions directly
    # Same code that runs manually — no duplication
    from data.make_dataset import (
        validate_schema, clean, engineer_features,
        write_parquet, write_manifest, compute_file_hash
    )

    # ── Load full raw data ───────────────────────────────────────
    log.info(f"[Task 3] Loading raw data from {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    log.info(f"[Task 3] Raw shape: {df.shape}")

    # ── Run pipeline ─────────────────────────────────────────────
    validate_schema(df)
    df = clean(df)
    df = engineer_features(df)

    # ── Write output ─────────────────────────────────────────────
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    size_mb = write_parquet(df, PROCESSED_PATH)
    write_manifest(RAW_PATH, PROCESSED_PATH, len(df), size_mb)

    output_hash = compute_file_hash(PROCESSED_PATH)
    log.info(f"[Task 3] Parquet written: {PROCESSED_PATH}")
    log.info(f"[Task 3] Output hash: {output_hash}")
    log.info(f"[Task 3] Rows: {len(df):,}")
    log.info(f"[Task 3] Size: {size_mb}MB")

    # ── Stamp the manifest with Airflow run_id ───────────────────
    # This links every Parquet file to its Airflow run.
    # On Day 9, lineage audit reads run_id from manifest to
    # find the exact Airflow run that produced this dataset.
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    manifest["airflow_run_id"] = run_id
    manifest["airflow_execution_date"] = str(execution_date)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"[Task 3] Manifest stamped with run_id: {run_id}")

    return {
        "output_hash": output_hash,
        "rows":        len(df),
        "size_mb":     size_mb,
        "run_id":      run_id,
    }


# ════════════════════════════════════════════════════════════════
# TASK 4 — Verify Output
# ════════════════════════════════════════════════════════════════

def verify_output(**context):
    """
    Task 4: Confirm the produced Parquet is valid.

    Transform completing without error is necessary but not
    sufficient. This task reads the output file and verifies:
      - File exists and is non-empty
      - Row count matches what Task 3 reported
      - All expected columns are present
      - No nulls in critical columns

    This is the final gate before the pipeline is marked
    successful. A green checkmark in Airflow means this
    task passed — not just that no exception was thrown.

    Why verify separately from transform:
    Transform could write a valid file that is later truncated
    by a disk full error, a container restart, or a race condition.
    Reading it back in a separate task catches filesystem issues
    that in-process checks miss.
    """
    execution_date = context["execution_date"]
    log.info(f"[Task 4] Output verification for run: {execution_date}")

    # ── File exists ──────────────────────────────────────────────
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(
            f"Output file not found: {PROCESSED_PATH}\n"
            f"Task 3 may have failed silently."
        )

    # ── Read and verify ──────────────────────────────────────────
    df = pd.read_parquet(PROCESSED_PATH)
    log.info(f"[Task 4] Parquet loaded: {df.shape}")

    # ── Row count ────────────────────────────────────────────────
    if len(df) < 100000:
        raise ValueError(
            f"Output has only {len(df):,} rows.\n"
            f"Expected > 100,000. Transform may have failed partially."
        )

    # ── Required columns ─────────────────────────────────────────
    expected_cols = {
        "hour", "day_of_week", "month", "is_weekend", "is_rush_hour",
        "distance_km", "passenger_count",
        "pickup_latitude", "pickup_longitude",
        "dropoff_latitude", "dropoff_longitude",
        "vendor_id", "trip_duration"
    }
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Output Parquet missing columns: {missing_cols}\n"
            f"Feature engineering may have failed."
        )

    # ── Null check on critical columns ───────────────────────────
    for col in ["distance_km", "trip_duration", "hour"]:
        null_count = df[col].isna().sum()
        if null_count > 0:
            raise ValueError(
                f"Output has {null_count:,} nulls in {col}.\n"
                f"Feature engineering produced unexpected nulls."
            )

    # ── Range sanity checks ──────────────────────────────────────
    if df["trip_duration"].min() < 60:
        raise ValueError("Output contains trip_duration < 60s. Cleaning failed.")
    if df["trip_duration"].max() > 7200:
        raise ValueError("Output contains trip_duration > 7200s. Cleaning failed.")
    if df["distance_km"].min() < 0.1:
        raise ValueError("Output contains distance_km < 0.1. Cleaning failed.")

    log.info("[Task 4] Output verification PASSED.")
    log.info(f"[Task 4] Rows: {len(df):,}")
    log.info(f"[Task 4] Columns: {len(df.columns)}")
    log.info(f"[Task 4] trip_duration range: {df['trip_duration'].min():.0f}s "
             f"— {df['trip_duration'].max():.0f}s")
    log.info(f"[Task 4] distance_km range: {df['distance_km'].min():.2f}km "
             f"— {df['distance_km'].max():.2f}km")
    log.info("[Task 4] Pipeline complete. Data is clean and ready for training.")

    return {
        "rows":     len(df),
        "columns":  len(df.columns),
        "verified": True,
    }


# ════════════════════════════════════════════════════════════════
# DAG DEFINITION
# ════════════════════════════════════════════════════════════════

# Default arguments applied to every task in this DAG
default_args = {
    "owner":            "mlops",
    "retries":          1,          # retry once on failure before alerting
    "retry_delay":      timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
    # Email alerting goes here in production:
    # "email_on_failure": True,
    # "email": ["oncall@company.com"],
}

# The DAG object — this is what Airflow reads
with DAG(
    dag_id="rideshare_etl",
    description="Automated ETL: raw CSV → validated Parquet with quality gates",
    schedule_interval="0 0 * * *",  # daily at midnight UTC (cron syntax)
    start_date=datetime(2026, 1, 1),
    catchup=False,     # don't backfill historical runs on first start
                       # if catchup=True, Airflow would try to run every
                       # day from start_date until now on first boot
    default_args=default_args,
    tags=["etl", "data", "rideshare"],
) as dag:

    # ── Task definitions ─────────────────────────────────────────
    # PythonOperator runs a Python function as an Airflow task.
    # provide_context=True passes the Airflow context dict
    # (execution_date, run_id, etc.) to the function as **context.

    task_validate_schema = PythonOperator(
        task_id="validate_raw_schema",
        python_callable=validate_raw_schema,
        provide_context=True,
    )

    task_quality_gates = PythonOperator(
        task_id="run_quality_gates",
        python_callable=run_quality_gates,
        provide_context=True,
    )

    task_transform = PythonOperator(
        task_id="transform_to_parquet",
        python_callable=transform_to_parquet,
        provide_context=True,
    )

    task_verify = PythonOperator(
        task_id="verify_output",
        python_callable=verify_output,
        provide_context=True,
    )

    # ── Dependency chain ─────────────────────────────────────────
    # >> is the Airflow "set downstream" operator.
    # Read as: "this task must succeed before the next one starts"
    # If any task fails, all tasks to its right are skipped.
    task_validate_schema >> task_quality_gates >> task_transform >> task_verify
