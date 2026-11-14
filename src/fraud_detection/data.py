"""Obtención del dataset y validación del esquema."""

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
    """Devuelve la raíz del proyecto a partir de este módulo fuente instalado."""
    return Path(__file__).resolve().parents[2]


def raw_data_path() -> Path:
    """Devuelve la ruta local del repositorio para el CSV de origen ignorado."""
    return repository_root() / "data" / "raw" / "creditcard.csv"


def ensure_dataset(path: str | Path | None = None) -> Path:
    """Descarga el dataset fijado solo cuando el CSV local no existe."""
    destination = Path(path) if path is not None else raw_data_path()
    if destination.is_file():
        return destination

    import kagglehub

    cached_directory = Path(kagglehub.dataset_download(DATASET_HANDLE))
    cached_csv = cached_directory / "creditcard.csv"
    if not cached_csv.is_file():
        raise FileNotFoundError(
            "El dataset fijado de Kaggle no contenía creditcard.csv."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_csv, destination)
    return destination


def validate_credit_card_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Valida las columnas esperadas, las entradas numéricas y las etiquetas binarias no nulas."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            "Se esperaba el dataset de tarjetas de crédito como un DataFrame de pandas."
        )

    actual_columns = tuple(frame.columns)
    if actual_columns != EXPECTED_COLUMNS:
        missing = [column for column in EXPECTED_COLUMNS if column not in actual_columns]
        unexpected = [column for column in actual_columns if column not in EXPECTED_COLUMNS]
        raise ValueError(
            "Columnas del dataset inesperadas o en orden incorrecto. "
            f"Faltantes: {missing}; inesperadas: {unexpected}; "
            f"orden esperado: {list(EXPECTED_COLUMNS)}."
        )

    non_numeric = [
        column
        for column in FEATURE_COLUMNS
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    if non_numeric:
        raise TypeError(f"Las columnas de entrada deben ser numéricas: {non_numeric}.")

    target = frame[TARGET_COLUMN]
    if not pd.api.types.is_numeric_dtype(target.dtype):
        raise TypeError("El objetivo Class debe ser numérico con valores 0 y 1.")
    if target.isna().any() or not target.isin((0, 1)).all():
        raise ValueError(
            "El objetivo Class no debe tener valores nulos y solo debe contener 0 y 1."
        )

    return frame


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Carga el CSV local o el dataset fijado de Kaggle y valida su esquema."""
    source_path = ensure_dataset(path)
    return validate_credit_card_schema(pd.read_csv(source_path))
