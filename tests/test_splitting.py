import numpy as np
import pandas as pd

from fraud_detection.splitting import create_grouped_partitions, grouped_overlap_count


def make_repeated_group_data() -> tuple[pd.DataFrame, pd.Series]:
    group_values = np.arange(50, dtype=float)
    unique_features = pd.DataFrame(
        {
            "Time": group_values,
            "Amount": group_values * 2,
        }
    )
    unique_target = pd.Series((group_values < 10).astype(int), name="Class")

    features = pd.concat([unique_features, unique_features], ignore_index=True)
    target = pd.concat([unique_target, unique_target], ignore_index=True)
    return features, target


def test_grouped_partitions_keep_duplicate_features_together_and_stratify():
    features, target = make_repeated_group_data()

    partitions = create_grouped_partitions(features, target)

    assert (
        len(partitions.x_fit)
        + len(partitions.x_validation)
        + len(partitions.x_test)
        == len(features)
    )
    assert grouped_overlap_count(partitions.groups_fit, partitions.groups_validation) == 0
    assert grouped_overlap_count(partitions.groups_fit, partitions.groups_test) == 0
    assert grouped_overlap_count(partitions.groups_validation, partitions.groups_test) == 0
    assert abs(partitions.y_fit.mean() - target.mean()) < 0.1
    assert abs(partitions.y_validation.mean() - target.mean()) < 0.1
    assert abs(partitions.y_test.mean() - target.mean()) < 0.1
