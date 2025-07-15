import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tempfile
import joblib
from typing import Any, Optional
import numpy as np


class Trainer:
    """
    Trainer class for fitting machine learning models, evaluating performance,
    logging metrics and artifacts to Weights & Biases (W&B), and saving outputs.
    """

    def __init__(
        self,
        model: Any,
        model_name: str,
        sensor_config: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        wandb_project: str = "sensor-fusion-har",
        log_to_wandb: bool = True,
        save_path: str = "models/"
    ) -> None:
        """
        Initialize the Trainer object.

        Args:
            model (Any): A model that implements the `train`, `evaluate`, and `predict` methods.
            model_name (str): Identifier for the model (e.g., 'xgb', 'logistic').
            sensor_config (str): Description of the sensor data configuration (e.g., 'fused', 'gyro').
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_test (np.ndarray): Testing features.
            y_test (np.ndarray): Testing labels.
            wandb_project (str): W&B project name.
            log_to_wandb (bool): Whether to enable W&B logging.
            save_path (str): Path to save models (not used if W&B artifact logging is enabled).
        """
        self.model = model
        self.model_name = model_name
        self.sensor_config = sensor_config

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.log_to_wandb = log_to_wandb
        self.save_path = Path(save_path)
        self.run: Optional[wandb.sdk.wandb_run.Run] = None

        if self.log_to_wandb:
            self.run = wandb.init(
                project=wandb_project,
                config={
                    "model": model_name,
                    "sensor_config": sensor_config
                },
                name=f"{model_name}_{sensor_config}",
                reinit=True
            )

    def train_and_evaluate(self) -> None:
        """
        Train the model, evaluate it, and optionally log results and save the model.
        """
        self.model.train(self.X_train, self.y_train)
        results = self.model.evaluate(self.X_train, self.y_train, self.X_test, self.y_test)

        if self.log_to_wandb:
            self._log_metrics(results)
            self._log_confusion_matrix(results["confusion_matrix"])
            self._save_model()

        if self.run:
            self.run.finish()

    def _save_model(self) -> None:
        """
        Save the trained model as a Weights & Biases artifact.
        """
        if not self.log_to_wandb:
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp_file:
            joblib.dump(self.model.model, tmp_file.name)

            artifact = wandb.Artifact(
                name=f"{self.model_name}_{self.sensor_config}_model",
                type="model",
                description=f"{self.model_name} trained on {self.sensor_config} data",
                metadata={
                    "model": self.model_name,
                    "sensor_config": self.sensor_config
                }
            )
            artifact.add_file(tmp_file.name)
            wandb.log_artifact(artifact)

    def _log_metrics(self, results: dict) -> None:
        """
        Log evaluation metrics to W&B.

        Args:
            results (dict): Dictionary containing metrics such as accuracy and F1 scores.
        """
        wandb.log({
            "train_accuracy": results["train_accuracy"],
            "test_accuracy": results["test_accuracy"],
            "train_f1": results["train_f1"],
            "test_f1": results["test_f1"],
            "generalization_gap": results["generalization_gap"]
        })

    def _log_confusion_matrix(self, conf_matrix: np.ndarray) -> None:
        """
        Log a confusion matrix visualization to W&B.

        Args:
            conf_matrix (np.ndarray): Confusion matrix from evaluation.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        plot_path = "outputs/plots/conf_matrix.png"
        Path("outputs/plots").mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)

        wandb.log({"confusion_matrix": wandb.Image(plot_path)})
        plt.close()
