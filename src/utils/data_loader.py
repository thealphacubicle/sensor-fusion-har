from pathlib import Path
import polars as pl
from typing import Tuple

DATA_DIR = Path("data/processed")

def load_data() -> Tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
    """
    Load the preprocessed train/test datasets using Polars.

    Assumes:
    - Files `train.parquet` and `test.parquet` are located in `data/processed/`
    - Columns include sensor features, 'subject', 'activity_id', and 'activity'

    Returns:
    - X_train: Features from training data (pl.DataFrame)
    - X_test: Features from testing data (pl.DataFrame)
    - y_train: Activity labels from training data (pl.Series)
    - y_test: Activity labels from testing data (pl.Series)
    """
    train_df = pl.read_parquet(DATA_DIR / "train.parquet")
    test_df = pl.read_parquet(DATA_DIR / "test.parquet")

    # Drop metadata columns we don't want as features
    X_train = train_df.drop(["activity", "activity_id", "subject"])
    y_train = train_df["activity"]

    X_test = test_df.drop(["activity", "activity_id", "subject"])
    y_test = test_df["activity"]

    return X_train, X_test, y_train, y_test
