import os
import pandas as pd
import plotly.express as px
import streamlit as st
import wandb

ENTITY = os.getenv("WANDB_ENTITY", "")
PROJECT = "sensor-fusion-har"

@st.cache_data
def fetch_runs(entity: str = ENTITY, project: str = PROJECT) -> pd.DataFrame:
    try:
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}") if entity else api.runs(project)
        records = []
        for run in runs:
            model = run.config.get("model")
            sensor = run.config.get("sensor_config")
            records.append({
                "model": model,
                "sensor_config": sensor,
                "train_accuracy": run.summary.get("train_accuracy"),
                "test_accuracy": run.summary.get("test_accuracy"),
                "train_f1": run.summary.get("train_f1"),
                "test_f1": run.summary.get("test_f1"),
                "generalization_gap": run.summary.get("generalization_gap"),
                "Runtime": run.summary.get("Runtime"),
                "label": f"{model} ({sensor})"
            })
        return pd.DataFrame(records)
    except:
        df = pd.read_csv("wandb_export_2025-08-08T06_51_32.601-04_00.csv")
        df["label"] = df.apply(lambda row: f"{row['model']} ({row['sensor_config']})", axis=1)
        return df

def main():
    st.set_page_config(page_title="Sensor Fusion HAR Dashboard", layout="wide")
    st.title("Sensor Fusion HAR Dashboard")

    with st.spinner("Loading data..."):
        df = fetch_runs()

    if df.empty:
        st.info("No runs found.")
        return

    st.success(f"Loaded {len(df)} runs")

    # Model type selector
    available_models = df["model"].dropna().unique().tolist()
    selected_models = st.multiselect(
        "Select model(s) to display:", available_models, default=available_models
    )

    df_filtered = df[df["model"].isin(selected_models)]

    with st.spinner("Rendering charts..."):
        # 1. Accuracy and F1 per model
        st.subheader("📊 Accuracy and F1 Score by Model Type")
        model_metrics = df_filtered.groupby("model")[[
            "train_accuracy",
            "test_accuracy",
            "train_f1",
            "test_f1",
        ]].mean().reset_index()
        fig1 = px.bar(
            model_metrics.melt(id_vars="model"),
            x="model",
            y="value",
            color="variable",
            barmode="group",
            labels={"value": "Score", "variable": "Metric"},
            title="Average Accuracy & F1 by Model Type",
        )
        st.plotly_chart(fig1, use_container_width=True)

        # 2. Generalization gap
        st.subheader("📉 Generalization Gap")
        fig2 = px.bar(
            df_filtered,
            x="label",
            y="generalization_gap",
            color="model",
            labels={"label": "Model (Sensor)", "generalization_gap": "Train-Test Accuracy Gap"},
            title="Generalization Gap per Configuration",
        )
        st.plotly_chart(fig2, use_container_width=True)

        # 3. Accuracy by sensor config
        if "sensor_config" in df_filtered.columns:
            st.subheader("📦 Sensor Config Comparison")
            sensor_group = df_filtered.groupby("sensor_config")[[
                "test_accuracy",
                "test_f1",
            ]].mean().reset_index()
            fig3 = px.bar(
                sensor_group.melt(id_vars="sensor_config"),
                x="sensor_config",
                y="value",
                color="variable",
                barmode="group",
                labels={"value": "Score", "sensor_config": "Sensor Setup"},
                title="Performance by Sensor Configuration",
            )
            st.plotly_chart(fig3, use_container_width=True)

        # 4. Runtime vs. accuracy
        if "Runtime" in df_filtered.columns:
            st.subheader("⏱️ Runtime vs. Accuracy")
            fig4 = px.scatter(
                df_filtered,
                x="Runtime",
                y="test_accuracy",
                color="model",
                hover_data=["label"],
                labels={"Runtime": "Training Runtime (s)", "test_accuracy": "Test Accuracy"},
                title="Test Accuracy vs Training Runtime",
            )
            st.plotly_chart(fig4, use_container_width=True)

if __name__ == "__main__":
    main()