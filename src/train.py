"""
Training entrypoint — Day 3 stub.

What this does TODAY (Day 3):
- Proves the training container can find and run this module
- Establishes the CLI interface (--data-path, --model-output)
- Logs that it was reached successfully

What this becomes on Day 5:
- Loads rideshare data from /app/data
- Runs feature engineering
- Fits XGBoost model
- Writes model.pkl to /models/

The interface (flags, paths) stays identical — only the body changes.
"""

import logging
import click

# Configure logging — every production system logs to stdout
# so Docker and Kubernetes can capture it without extra setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--data-path",
    default="/app/data",
    help="Directory containing rideshare training data (CSVs)"
)
@click.option(
    "--model-output",
    default="/models/model.pkl",
    help="Full path where trained model artifact will be written"
)
def train(data_path: str, model_output: str):
    """
    Train the ridesharing demand predictor.
    Day 5 implements the actual training logic.
    """
    log.info("=== Rideshare Demand Predictor — Training ===")
    log.info(f"Data path   : {data_path}")
    log.info(f"Model output: {model_output}")
    log.info("Day 3 stub: container entrypoint reached successfully.")
    log.info("Day 5 will implement: load data → features → XGBoost → save artifact.")


if __name__ == "__main__":
    train()
