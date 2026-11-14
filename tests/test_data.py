import pandas as pd
import pytest

from fraud_detection.data import EXPECTED_COLUMNS, FEATURE_COLUMNS, validate_credit_card_schema


def make_valid_frame() -> pd.DataFrame:
    values = {column: [0.0, 1.0] for column in FEATURE_COLUMNS}
    values["Class"] = [0, 1]
    return pd.DataFrame(values, columns=EXPECTED_COLUMNS)


def test_schema_accepts_expected_numeric_inputs_and_binary_target():
    frame = make_valid_frame()

    validated = validate_credit_card_schema(frame)

    assert validated is frame
    assert tuple(validated.columns) == EXPECTED_COLUMNS


def test_schema_rejects_missing_or_reordered_columns():
    frame = make_valid_frame().drop(columns="Amount")

    with pytest.raises(ValueError, match="Columnas del dataset inesperadas"):
        validate_credit_card_schema(frame)


def test_schema_rejects_non_numeric_inputs():
    frame = make_valid_frame()
    frame["Time"] = ["first", "second"]

    with pytest.raises(TypeError, match="deben ser numéricas"):
        validate_credit_card_schema(frame)


@pytest.mark.parametrize("target_values", [[0, 2], [0, None]])
def test_schema_rejects_invalid_target_values(target_values):
    frame = make_valid_frame()
    frame["Class"] = target_values

    with pytest.raises(ValueError, match="no debe tener valores nulos"):
        validate_credit_card_schema(frame)
