import pandas as pd
from pathlib import Path
import urllib.request
import zipfile
import logging

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# === Paths ===
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ZIP_PATH = RAW_DIR / "HAR_wrapper.zip"
WRAPPER_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
DATASET_DIR = RAW_DIR / "UCI HAR Dataset"
NESTED_ZIP = RAW_DIR / "UCI HAR Dataset.zip"


# === Activity Mapping ===
def load_activity_map():
    logger.info("Loading activity map from %s", DATASET_DIR / "activity_labels.txt")
    df = pd.read_csv(DATASET_DIR / "activity_labels.txt", sep=r'\s+', header=None, names=["id", "label"])
    logger.debug("Activity map loaded: %s", df.head())
    return dict(zip(df["id"], df["label"]))


# === Feature Names (fixed for duplicates) ===
def load_feature_names():
    logger.info("Loading feature names from %s", DATASET_DIR / "features.txt")
    df = pd.read_csv(DATASET_DIR / "features.txt", sep=r'\s+', header=None, names=["index", "feature"])
    names = df["feature"].tolist()

    # Fix duplicates by adding suffixes
    seen = {}
    unique_names = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            unique_names.append(name)
        else:
            seen[name] += 1
            unique_names.append(f"{name}__{seen[name]}")
    logger.debug("Feature names loaded and deduplicated: %d features", len(unique_names))
    return unique_names


# === Downloader ===
def download_and_extract():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET_DIR.exists():
        logger.info("[✓] Dataset already exists at %s", DATASET_DIR)
        return

    logger.info("[⏳] Downloading dataset from %s", WRAPPER_URL)
    urllib.request.urlretrieve(WRAPPER_URL, ZIP_PATH)
    logger.info("[✓] Downloaded to %s", ZIP_PATH)

    logger.info("[📦] Extracting outer zip...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(RAW_DIR)

    if NESTED_ZIP.exists():
        logger.info("[📦] Extracting inner zip...")
        with zipfile.ZipFile(NESTED_ZIP, 'r') as zf:
            zf.extractall(RAW_DIR)
        NESTED_ZIP.unlink()

    ZIP_PATH.unlink()
    logger.info("[✓] Dataset ready at %s", DATASET_DIR)


# === Preprocessing ===
def load_split(split="train"):
    logger.info("Loading split: %s", split)
    feature_names = load_feature_names()
    activity_map = load_activity_map()

    X_path = DATASET_DIR / f"{split}/X_{split}.txt"
    y_path = DATASET_DIR / f"{split}/y_{split}.txt"
    subject_path = DATASET_DIR / f"{split}/subject_{split}.txt"

    logger.info("Reading features from %s", X_path)
    X = pd.read_csv(X_path, sep=r'\s+', header=None, names=feature_names)
    logger.info("Reading activities from %s", y_path)
    y = pd.read_csv(y_path, header=None, names=["activity_id"])
    logger.info("Reading subjects from %s", subject_path)
    subject = pd.read_csv(subject_path, header=None, names=["subject"])

    df = pd.concat([X, y, subject], axis=1)
    df["activity"] = df["activity_id"].map(activity_map)
    logger.info("Loaded split '%s' with shape %s", split, df.shape)
    return df


def preprocess_and_save():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Preprocessing train split")
    df_train = load_split("train")
    logger.info("Preprocessing test split")
    df_test = load_split("test")

    train_path = PROCESSED_DIR / "train.parquet"
    test_path = PROCESSED_DIR / "test.parquet"

    logger.info("Saving train split to %s", train_path)
    df_train.to_parquet(train_path, index=False)
    logger.info("Saving test split to %s", test_path)
    df_test.to_parquet(test_path, index=False)

    logger.info("[✓] Saved to data/processed/: train.parquet (%s), test.parquet (%s)", df_train.shape, df_test.shape)


# === Entry Point ===
if __name__ == "__main__":
    download_and_extract()
    preprocess_and_save()