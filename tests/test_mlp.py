import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from fraud_detection.mlp import (
    MLP_PARAMETER_GRID,
    build_mlp_pipeline,
    build_mlp_search,
    combine_development_partitions,
    evaluate_mlp_test,
)
from fraud_detection.splitting import GroupedPartitions


def test_mlp_pipeline_orders_scaling_and_smote_before_classifier():
    pipeline = build_mlp_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == [
        "scaler",
        "smote",
        "classifier",
    ]
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["smote"], SMOTE)
    assert pipeline.named_steps["smote"].random_state == 42

    classifier = pipeline.named_steps["classifier"]
    assert isinstance(classifier, MLPClassifier)
    assert classifier.activation == "relu"
    assert classifier.solver == "adam"
    assert classifier.batch_size == 256
    assert classifier.max_iter == 100
    assert classifier.early_stopping is False
    assert classifier.random_state == 42


def test_mlp_search_is_bounded_grouped_and_fails_on_estimator_errors():
    search = build_mlp_search()

    candidates = list(ParameterGrid(search.param_grid))
    assert len(MLP_PARAMETER_GRID) == 2
    assert len(candidates) == 2
    assert [candidate["classifier__hidden_layer_sizes"] for candidate in candidates] == [
        (32,),
        (64, 32),
    ]
    assert [candidate["classifier__alpha"] for candidate in candidates] == [
        1e-3,
        1e-4,
    ]
    assert all(candidate["classifier__learning_rate_init"] == 1e-3 for candidate in candidates)
    assert isinstance(search.cv, StratifiedGroupKFold)
    assert search.cv.n_splits == 3
    assert search.cv.shuffle is True
    assert search.cv.random_state == 44
    assert search.scoring == "average_precision"
    assert search.refit is True
    assert search.n_jobs == 1
    assert search.pre_dispatch == 1
    assert search.error_score == "raise"


def test_development_data_combines_only_inner_training_and_validation():
    partitions = GroupedPartitions(
        x_fit=pd.DataFrame({"marker": [1, 2]}, index=[10, 11]),
        y_fit=pd.Series([0, 1], index=[10, 11], name="Class"),
        x_validation=pd.DataFrame({"marker": [3]}, index=[12]),
        y_validation=pd.Series([0], index=[12], name="Class"),
        x_test=pd.DataFrame({"marker": [999]}, index=[13]),
        y_test=pd.Series([1], index=[13], name="Class"),
        groups_fit=np.array([100, 101]),
        groups_validation=np.array([102]),
        groups_test=np.array([999]),
    )

    features, target, groups = combine_development_partitions(partitions)

    assert features["marker"].tolist() == [1, 2, 3]
    assert target.tolist() == [0, 1, 0]
    assert groups.tolist() == [100, 101, 102]


class FixedPredictions:
    def predict_proba(self, features):
        return np.array(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.2, 0.8],
                [0.7, 0.3],
                [0.4, 0.6],
                [0.3, 0.7],
            ]
        )

    def predict(self, features):
        return np.array([0, 1, 1, 0, 1, 1])


def test_final_test_metrics_use_probability_scores_and_tn_fp_fn_tp_order():
    features = pd.DataFrame({"feature": range(6)})
    target = pd.Series([0, 0, 0, 1, 1, 1], name="Class")

    result = evaluate_mlp_test(FixedPredictions(), features, target)

    assert result.row_count == 6
    assert result.fraud_count == 3
    assert result.average_precision == pytest.approx(43 / 90)
    assert result.tn == 1
    assert result.fp == 2
    assert result.fn == 1
    assert result.tp == 2
    assert list(result.preview.columns) == [
        "actual_label",
        "fraud_probability",
        "predicted_label",
    ]
    assert len(result.preview) == 6
