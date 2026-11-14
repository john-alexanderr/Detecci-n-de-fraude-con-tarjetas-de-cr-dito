"""Dataset retrieval and schema validation."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

DATASET_HANDLE = "mlg-ulb/creditcardfraud/versions/3"
FEATURE_COLUMNS = (
    "Time",
    *(f"V{index}" for index in range(1, 29)),
    "Amount",
)
TARGET_COLUMN = "Class"
EXPECTED_COLUMNS = (*FEATURE_COLUMNS, TARGET_COLUMN)


def repository_root() -> Path:
    """Return the project root based on this installed source module."""
    return Path(__file__).resolve().parents[2]


def raw_data_path() -> Path:
    """Return the repository-local path for the ignored source CSV."""
    return repository_root() / "data" / "raw" / "creditcard.csv"


def ensure_dataset(path: str | Path | None = None) -> Path:
    """Download the pinned dataset only when the local CSV is absent."""
    destination = Path(path) if path is not None else raw_data_path()
    if destination.is_file():
        return destination

    import kagglehub

    cached_directory = Path(kagglehub.dataset_download(DATASET_HANDLE))
    cached_csv = cached_directory / "creditcard.csv"
    if not cached_csv.is_file():
        raise FileNotFoundError(
            "The pinned Kaggle dataset did not contain creditcard.csv."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_csv, destination)
    return destination


def validate_credit_card_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate expected columns, numeric inputs, and binary non-null labels."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected the credit-card dataset as a pandas DataFrame.")

    actual_columns = tuple(frame.columns)
    if actual_columns != EXPECTED_COLUMNS:
        missing = [column for column in EXPECTED_COLUMNS if column not in actual_columns]
        unexpected = [column for column in actual_columns if column not in EXPECTED_COLUMNS]
        raise ValueError(
            "Unexpected dataset columns or order. "
            f"Missing: {missing}; unexpected: {unexpected}; "
            f"expected order: {list(EXPECTED_COLUMNS)}."
        )

    non_numeric = [
        column
        for column in FEATURE_COLUMNS
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    if non_numeric:
        raise TypeError(f"Input columns must be numeric: {non_numeric}.")

    target = frame[TARGET_COLUMN]
    if not pd.api.types.is_numeric_dtype(target.dtype):
        raise TypeError("The Class target must be numeric with values 0 and 1.")
    if target.isna().any() or not target.isin((0, 1)).all():
        raise ValueError("The Class target must be non-null and contain only 0 and 1.")

    return frame


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the local or pinned Kaggle CSV and validate its schema."""
    source_path = ensure_dataset(path)
    return validate_credit_card_schema(pd.read_csv(source_path))
