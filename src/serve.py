"""
Inference entrypoint — Day 3 stub.

What this does TODAY (Day 3):
- Proves the inference container starts cleanly
- Reads MODEL_PATH environment variable
- Fails loudly (not silently) if MODEL_PATH is missing
- Logs that it was reached successfully

What this becomes on Day 16:
- Loads model.pkl from MODEL_PATH at startup
- Exposes POST /predict endpoint via FastAPI
- Accepts JSON: {"location": "5th & Main", "hour": 18, "day": "Friday"}
- Returns JSON: {"demand_score": 0.87, "confidence": 0.91}
- Runs uvicorn on port 8000

The MODEL_PATH contract stays identical — only the body changes.

Why we fail loudly on missing MODEL_PATH:
- Silent failures (serving with no model) are more dangerous than crashes
- Kubernetes will restart a crashed container and alert you
- Kubernetes cannot detect a container that starts and serves garbage
- This is the core lesson of docs/silent_failure.md (Day 5)
"""

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def load_model(model_path: str):
    """
    Load model artifact from model_path.
    Today: validates the path exists and logs it.
    Day 16: deserializes model.pkl with joblib and returns it.
    """
    log.info(f"Attempting to load model from: {model_path}")

    # Day 16 replaces this block with:
    # import joblib
    # model = joblib.load(model_path)
    # return model

    log.info("[STUB] Model load skipped — Day 16 implements this.")
    return None


def main():
    # Read MODEL_PATH from environment — never hardcode this
    # Set it at runtime: docker run -e MODEL_PATH=/models/model.pkl
    model_path = os.environ.get("MODEL_PATH", "").strip()

    if not model_path:
        # Loud failure — better than silent garbage
        log.error("FATAL: MODEL_PATH environment variable is not set.")
        log.error("Set it with: docker run -e MODEL_PATH=/models/model.pkl")
        log.error("This container cannot serve predictions without a model.")
        sys.exit(1)  # Non-zero exit = Kubernetes knows something is wrong

    model = load_model(model_path)

    log.info("=== Rideshare Demand Predictor — Inference ===")
    log.info(f"Model path  : {model_path}")
    log.info(f"Model loaded: {model is not None}")
    log.info("Day 3 stub: inference container alive and waiting.")
    log.info("Day 16 will implement: FastAPI + /predict endpoint on port 8000.")


if __name__ == "__main__":
    main()
