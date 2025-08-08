import yaml
import polars as pl
import logging
from concurrent.futures import ThreadPoolExecutor
from src.models.LogisticRegression import LogisticRegressionModel
from src.models.RandomForest import RFClassifierModel
from src.models.KNN import KNNModel
from src.models.SVM import SVMModel
from src.models.NaiveBayes import NaiveBayesModel
from src.utils.trainer import Trainer, train_single_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Map string class names from YAML to actual classes
MODEL_CLASS_MAP = {
    "LogisticRegressionModel": LogisticRegressionModel,
    "KNNModel": KNNModel,
    "RandomForestClassifier": RFClassifierModel,
    "GaussianNBModel": NaiveBayesModel,
    "SVMModel": SVMModel
}


def get_model_instance(class_name: str, params: dict = None):
    """
    Instantiates the model class dynamically using provided parameters.
    """
    model_cls = MODEL_CLASS_MAP.get(class_name)
    if not model_cls:
        logging.error(f"Unknown model class: {class_name}")
        raise ValueError(f"Unknown model class: {class_name}")
    logging.info(f"Instantiating model: {class_name} with params: {params}")
    return model_cls(**params) if params else model_cls()


def load_yaml(path: str) -> dict:
    """
    Loads a YAML configuration file.
    """
    logging.info(f"Loading YAML config from: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # === Load Configs ===
    logging.info("Loading model and training configs...")
    models_config = load_yaml("config/models.yaml")
    training_config = Trainer.load_training_config("config/training_config.yaml")

    notify_config = training_config.get("notify", {"send": False, "type": "desktop"})
    max_workers = training_config.get("max_workers", 2)

    models = [model_name for model_name in models_config["models"]
              if models_config["models"][model_name].get("enabled", False)]

    logging.info(f"Trainable models: {models}")

    sensors = models_config["sensors"]
    logging.info(f"Trainable sensors: {sensors}")


    # === Load Data ===
    logging.info("Loading training and test data...")
    train_df = pl.read_parquet("data/processed/train.parquet")
    test_df = pl.read_parquet("data/processed/test.parquet")

    X_train = train_df.drop(["activity", "subject"])
    y_train = train_df["activity"]
    X_test = test_df.drop(["activity", "subject"])
    y_test = test_df["activity"]

    # === Schedule Training Jobs ===
    jobs = []

    logging.info("Scheduling training jobs...")
    for model_name in models:
        details = models_config["models"][model_name]
        class_name = details["class"]
        params = details.get("params", {})

        for sensor in sensors:
            logging.info(f"Preparing job for model: {model_name}, sensor: {sensor}")
            try:
                model_instance = get_model_instance(class_name, params)
            except Exception as exc:
                logging.exception(
                    f"Failed to instantiate model {model_name} with params {params}: {exc}"
                )
                continue

            trainer_args = (
                model_instance,
                model_name,
                sensor,
                X_train,
                y_train,
                X_test,
                y_test,
            )

            jobs.append((trainer_args, notify_config))

    # === Execute in Parallel ===
    logging.info(f"Executing {len(jobs)} jobs in parallel with max_workers={max_workers}...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(lambda args: train_single_model(*args), jobs)

    logging.info("All training jobs completed.")


if __name__ == "__main__":
    main()
