

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from src.feature_engineering.feature_manifest import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)


def build_preprocessor(
    scale_numerical: bool = False,
) -> ColumnTransformer:
    numerical = [c for c in NUMERICAL_FEATURES]

    string_categorical = [
        c for c in ("subject", "topic") if c in CATEGORICAL_FEATURES
    ]

    integer_categorical = [
        c for c in ("hour_of_day", "day_of_week")
        if c in CATEGORICAL_FEATURES
    ]

    boolean = list(BOOLEAN_FEATURES)

    if scale_numerical:
        numerical_steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    else:
        numerical_steps = [
            ("imputer", SimpleImputer(strategy="median")),
        ]

    transformers = [
        (
            "numerical",
            Pipeline(numerical_steps),
            numerical,
        ),
        (
            "string_categorical",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]
            ),
            string_categorical,
        ),
        (
            "integer_categorical",
            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]
            ),
            integer_categorical,
        ),
        (
            "boolean",
            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value=0),
                    ),
                ]
            ),
            boolean,
        ),
    ]

    return ColumnTransformer(transformers=transformers)
