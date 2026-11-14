import pytest

from fraud_detection.evaluation import summarize_predictions


def test_metrics_use_binary_labels_and_report_default_threshold_confusion_counts():
    metrics = summarize_predictions(
        y_true=[0, 0, 1, 1],
        y_score=[0.1, 0.9, 0.8, 0.2],
        y_pred=[0, 1, 1, 0],
    )

    assert metrics["average_precision"] == pytest.approx(7 / 12)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f2"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
