import os
from typing import List
import pandas as pd
import plotly.express as px
import streamlit as st
import wandb

# Constants for W&B project.
# The entity (username or team) can be overridden via the WANDB_ENTITY env var.
ENTITY = os.getenv("WANDB_ENTITY", "")
PROJECT = "sensor-fusion-har"


@st.cache_data
def fetch_runs(entity: str = ENTITY, project: str = PROJECT) -> pd.DataFrame:
    """Return a DataFrame with summary metrics for all runs."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}") if entity else api.runs(project)
    records: List[dict] = []
    for run in runs:
        records.append(
            {
                "run": run.name,
                "train_accuracy": run.summary.get("train_accuracy"),
                "test_accuracy": run.summary.get("test_accuracy"),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    """Render the Streamlit dashboard."""
    st.title("Sensor Fusion HAR Metrics")
    df = fetch_runs()
    if df.empty:
        st.info("No runs found.")
        return
    st.write(f"Loaded {len(df)} W&B runs")
    fig = px.line(
        df,
        x="run",
        y=["train_accuracy", "test_accuracy"],
        markers=True,
        title="Accuracy across runs",
        labels={"run": "Run", "value": "Accuracy", "variable": "Metric"},
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()