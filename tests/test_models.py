from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from fraud_detection.models import build_baseline_pipelines


def test_baseline_pipelines_keep_scaling_and_smote_inside_training_pipeline():
    pipelines = build_baseline_pipelines()

    assert list(pipelines) == ["Random Forest", "AdaBoost"]
    for pipeline in pipelines.values():
        assert [name for name, _ in pipeline.steps] == [
            "scaler",
            "smote",
            "classifier",
        ]
        assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
        assert isinstance(pipeline.named_steps["smote"], SMOTE)
        assert pipeline.named_steps["smote"].random_state == 42

    random_forest = pipelines["Random Forest"].named_steps["classifier"]
    assert isinstance(random_forest, RandomForestClassifier)
    assert random_forest.n_estimators == 200
    assert random_forest.random_state == 42
    assert random_forest.n_jobs == -1
