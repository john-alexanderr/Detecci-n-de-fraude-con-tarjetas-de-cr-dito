"""Leak-safe baseline pipeline definitions."""

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def build_baseline_pipelines() -> dict[str, Pipeline]:
    """Create the prior baseline estimators in scaler-SMOTE-model order."""
    classifiers = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "AdaBoost": AdaBoostClassifier(random_state=42),
    }
    return {
        name: Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=42)),
                ("classifier", classifier),
            ]
        )
        for name, classifier in classifiers.items()
    }
