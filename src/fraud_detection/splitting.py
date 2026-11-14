"""Duplicate-group-aware outer and inner validation partitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

OUTER_RANDOM_STATE = 42
INNER_RANDOM_STATE = 43
N_SPLITS = 5


@dataclass(frozen=True)
class GroupedPartitions:
    """Train, validation, and reserved test partitions with group labels."""

    x_fit: pd.DataFrame
    y_fit: pd.Series
    x_validation: pd.DataFrame
    y_validation: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    groups_fit: np.ndarray
    groups_validation: np.ndarray
    groups_test: np.ndarray


def feature_group_ids(features: pd.DataFrame) -> np.ndarray:
    """Hash input features only so identical feature rows share a group."""
    return pd.util.hash_pandas_object(features, index=False).to_numpy(dtype="uint64")


def grouped_overlap_count(left: np.ndarray, right: np.ndarray) -> int:
    """Count distinct group identifiers present in both partitions."""
    return int(np.intersect1d(np.unique(left), np.unique(right)).size)


def _require_disjoint_groups(left: np.ndarray, right: np.ndarray, name: str) -> None:
    overlap_count = grouped_overlap_count(left, right)
    if overlap_count:
        raise RuntimeError(f"{name} has {overlap_count} overlapping feature groups.")


def create_grouped_partitions(
    features: pd.DataFrame,
    target: pd.Series,
) -> GroupedPartitions:
    """Create reproducible stratified group holdouts without dropping rows."""
    if len(features) != len(target):
        raise ValueError("Features and target must contain the same number of rows.")
    if not features.index.equals(target.index):
        raise ValueError("Features and target must have the same row index.")

    groups = feature_group_ids(features)
    outer_splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=OUTER_RANDOM_STATE,
    )
    development_indices, test_indices = next(
        outer_splitter.split(features, target, groups=groups)
    )

    x_development = features.iloc[development_indices].copy()
    y_development = target.iloc[development_indices].copy()
    groups_development = groups[development_indices]
    groups_test = groups[test_indices]
    _require_disjoint_groups(groups_development, groups_test, "Outer split")

    inner_splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=INNER_RANDOM_STATE,
    )
    fit_indices, validation_indices = next(
        inner_splitter.split(
            x_development,
            y_development,
            groups=groups_development,
        )
    )

    groups_fit = groups_development[fit_indices]
    groups_validation = groups_development[validation_indices]
    _require_disjoint_groups(groups_fit, groups_validation, "Inner split")
    _require_disjoint_groups(groups_fit, groups_test, "Training and outer test")
    _require_disjoint_groups(groups_validation, groups_test, "Validation and outer test")

    return GroupedPartitions(
        x_fit=x_development.iloc[fit_indices].copy(),
        y_fit=y_development.iloc[fit_indices].copy(),
        x_validation=x_development.iloc[validation_indices].copy(),
        y_validation=y_development.iloc[validation_indices].copy(),
        x_test=features.iloc[test_indices].copy(),
        y_test=target.iloc[test_indices].copy(),
        groups_fit=groups_fit,
        groups_validation=groups_validation,
        groups_test=groups_test,
    )
