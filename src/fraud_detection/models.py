"""Definiciones de pipelines base sin fuga de información."""

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def build_baseline_pipelines() -> dict[str, Pipeline]:
    """Crea los estimadores base de referencia en orden escalador, SMOTE y clasificador."""
    classifiers = {
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
