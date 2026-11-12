from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core validation: column completeness & data quality
# ---------------------------------------------------------------------------

# Reference TEST_SIZE and RANDOM_SEED from original src/utils/config
from dependencies.utils.config import TEST_SIZE, RANDOM_SEED

def validate_features(
    df: pd.DataFrame,
    required: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Validate that the DataFrame contains all required features + target.

    Mirrors src/feature_engineering/validators.validate_features logic.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame to validate.
    required : List[str] | None
        Additional required column names beyond ALL_FEATURES + TARGET.

    Returns
    -------
    Dict[str, Any]
        Validation report with status, missing columns, null percentages,
        and type warnings.
    """
    from dependencies.feature_engineering.feature_manifest import ALL_FEATURES, TARGET

    required_all = (ALL_FEATURES + [TARGET]) if required is None else (
        list(ALL_FEATURES) + [TARGET] + list(required)
    )

    missing = set(required_all) - set(df.columns)
    extra = set(df.columns) - set(required_all)

    report: Dict[str, Any] = {
        "valid": len(missing) == 0,
        "missing_columns": sorted(missing),
        "extra_columns": sorted(extra),
        "null_summary": {},
        "type_warnings": [],
    }

    if missing:
        report["valid"] = False

    # Null percentage per column
    for col in df.columns:
        null_pct = float(df[col].isna().mean()) * 100.0
        if null_pct > 0:
            report["null_summary"][col] = round(null_pct, 2)

    # Type warnings for key columns
    for col in ALL_FEATURES:
        if col in df.columns:
            if df[col].dtype == object and col not in CATEGORICAL_FEATURES:
                report["type_warnings"].append(
                    f"Column '{col}' is object type; expected numerical or categorical."
                )

    return report


# Pull categorical features list for type-checking
from dependencies.feature_engineering.feature_manifest import CATEGORICAL_FEATURES  # noqa: E402


# ---------------------------------------------------------------------------
# MLflow-integrated validation logging (Databricks free tier)
# ---------------------------------------------------------------------------

def log_validation_to_mlflow(
    report: Dict[str, Any],
    run_name: str = "feature-validation",
) -> None:
    """
    Log validation report metrics to MLflow.

    On Databricks free tier, MLflow tracking URI defaults to the cluster
    environment. This function tags the run and logs null-count metrics.

    Parameters
    ----------
    report : Dict[str, Any]
        Validation report dict from validate_features().
    run_name : str
        MLflow run name prefix.
    """
    import mlflow

    with mlflow.start_run(run_name=f"{run_name}_{id(report)}") as run:
        # Log validation status
        mlflow.set_tag("validation_valid", str(report["valid"]))
        mlflow.set_tag("missing_columns", str(report["missing_columns"]))
        mlflow.set_tag("extra_columns", str(report["extra_columns"]))

        # Log null percentages as metrics
        for col, pct in report.get("null_summary", {}).items():
            mlflow.log_metric(f"null_pct_{col}", pct)

        # Log type warning count
        mlflow.log_metric(
            "type_warning_count",
            len(report.get("type_warnings", [])),
        )

        print(
            f"MLflow validation run {run.info.run_id}: "
            f"valid={report['valid']}, "
            f"null_cols={len(report.get('null_summary', {}))}"
        )


# ---------------------------------------------------------------------------
# Data split validation (stratify check)
# ---------------------------------------------------------------------------

def validate_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    stratify: bool = True,
) -> Dict[str, Any]:
    """
    Validate that train/test split maintains positive rate and stratification.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float
        Fraction of data for testing.
    stratify : bool
        Whether to stratify by target.

    Returns
    -------
    Dict[str, Any]
        Split validation report.
    """
    from sklearn.model_selection import train_test_split as _split
    import pandas as pd

    y = pd.Series(y) if not isinstance(y, pd.Series) else y
    X_train, X_test, y_train, y_test = _split(
        X, y, test_size=test_size, stratify=y if stratify else None, random_state=RANDOM_SEED
    )

    y_train_series = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train
    y_test_series = pd.Series(y_test) if not isinstance(y_test, pd.Series) else y_test

    pos_rate_full = float(y.mean()) if len(y) > 0 else 0.0
    pos_rate_train = float(y_train_series.mean()) if len(y_train_series) > 0 else 0.0
    pos_rate_test = float(y_test_series.mean()) if len(y_test_series) > 0 else 0.0

    return {
        "valid": True,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "positive_rate_full": round(pos_rate_full, 4),
        "positive_rate_train": round(pos_rate_train, 4),
        "positive_rate_test": round(pos_rate_test, 4),
        "stratification_deviation": round(
            abs(pos_rate_train - pos_rate_test), 4
        ),
    }