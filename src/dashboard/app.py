import os
from typing import List

import pandas as pd
import plotly.express as px
import reflex as rx
import wandb

# Constants for W&B project.
# The entity (username or team) can be overridden via the WANDB_ENTITY env var.
ENTITY = os.getenv("WANDB_ENTITY", "")
PROJECT = "sensor-fusion-har"


def fetch_runs(entity: str = ENTITY, project: str = PROJECT) -> pd.DataFrame:
    """Return a DataFrame with summary metrics for all runs.

    Parameters
    ----------
    entity: str
        W&B entity (user or team). Defaults to ``WANDB_ENTITY`` env var or the
        default associated with the API key.
    project: str
        The W&B project name to query.
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}") if entity else api.runs(project)
    records: List[dict] = []
    for run in runs:
        records.append({
            "run": run.name,
            "train_accuracy": run.summary.get("train_accuracy"),
            "test_accuracy": run.summary.get("test_accuracy"),
        })
    return pd.DataFrame(records)


def accuracy_chart(df: pd.DataFrame):
    """Create a line chart comparing train and test accuracy."""
    if df.empty:
        return rx.text("No runs found.")
    fig = px.line(
        df,
        x="run",
        y=["train_accuracy", "test_accuracy"],
        markers=True,
        title="Accuracy across runs",
        labels={"run": "Run", "value": "Accuracy", "variable": "Metric"},
    )
    return rx.plotly(data=fig.data, layout=fig.layout)


def dashboard() -> rx.Component:
    df = fetch_runs()
    return rx.vstack(
        rx.heading("Sensor Fusion HAR Metrics"),
        rx.text(f"Loaded {len(df)} W&B runs"),
        accuracy_chart(df),
        spacing="2em",
        padding="2em",
    )


app = rx.App()
app.add_page(dashboard, title="Sensor Fusion HAR Dashboard")
