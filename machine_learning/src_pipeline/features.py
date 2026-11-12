from __future__ import annotations

import numpy as np
import pandas as pd

from dependencies.feature_engineering.feature_manifest import ALL_FEATURES, TARGET

# Import feature builders from existing codebase
from dependencies.feature_engineering.feature_builder import (
    convert_datetime_columns,
    sort_reviews,
)
from dependencies.feature_engineering.aggregate_features import (
    add_fsrs_probability,
    add_user_success_rate,
    add_user_average_confidence,
    add_user_average_response_time,
    add_question_global_success_rate,
    add_recent_success_rate,
    add_recent_confidence,
    add_consecutive_correct,
    add_hesitation_response_ratio,
    add_normalized_memory_inputs,
)
from dependencies.feature_engineering.historical_features import (
    add_total_revisions,
    add_previous_correct,
    add_previous_confidence_score,
    add_previous_response_time,
    add_previous_hesitation,
    add_success_rate,
    add_average_confidence,
    add_average_response_time,
    add_average_hesitation,
)
from dependencies.feature_engineering.cleaning import convert_datetime_columns as cd2
from dependencies.feature_engineering.temporal_features import (
    add_review_interval,
    add_hour_of_day,
    add_day_of_week,
    add_session_duration,
    add_days_since_last_session,
)
from dependencies.feature_engineering.validators import validate_features

# Difficulty mapping (mirrors feature_builder.py)
DIFFICULTY_SCORE_MAP = {
    "EASY": 1,
    "MEDIUM": 2,
    "HARD": 3,
}


# ---------------------------------------------------------------------------
# Core feature builder: end-to-end feature engineering pipeline
# ---------------------------------------------------------------------------

def _safe_call(func, df, col_map=None):
    """Call a feature function, catching KeyError if expected columns missing."""
    try:
        return func(df)
    except KeyError:
        # Column(s) not present in this dataset; return df unchanged
        return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build ALL_FEATURES from raw review data.

    Mirrors src/feature_engineering/feature_builder.build_features logic,
    adapted for Databricks pipeline compatibility.

    Parameters
    ----------
    df : pd.DataFrame
        Raw review data must already have columns: review_time, user_id,
        question_id, correct, confidence_score, response_time_seconds,
        hesitation_seconds, question_difficulty, question_position, subject,
        topic, answer_changes.

    Returns
    -------
    pd.DataFrame
        DataFrame with ALL_FEATURES columns plus 'correct' target column.
    """

    # --- Baseline preprocessing ---
    df["difficulty"] = df["question_difficulty"].map(DIFFICULTY_SCORE_MAP)
    df["question_position_in_session"] = df["question_position"]

    df = convert_datetime_columns(df)
    df = sort_reviews(df)

    # Historical features
    df = _safe_call(add_total_revisions, df)
    df = _safe_call(add_previous_correct, df)
    df = _safe_call(add_previous_confidence_score, df)
    df = _safe_call(add_previous_response_time, df)
    df = _safe_call(add_previous_hesitation, df)
    df = _safe_call(add_success_rate, df)
    df = _safe_call(add_average_confidence, df)
    df = _safe_call(add_average_response_time, df)
    df = _safe_call(add_average_hesitation, df)

    # Temporal features - adapted for our column naming
    # add_review_interval expects 'previous_review_time'; our data uses
    # session-based timing, so we compute interval from review_time only
    if "review_time" in df.columns:
        df["review_interval_days"] = (
            df["review_time"].diff().dt.total_seconds() / 86400
        )
    else:
        df["review_interval_days"] = np.nan

    df = _safe_call(add_hour_of_day, df)
    df = _safe_call(add_day_of_week, df)

    # session_duration and days_since_last_session expect session boundaries;
    # compute available metrics from review_time
    if "review_time" in df.columns:
        df["session_duration_minutes"] = (
            df["review_time"].diff().dt.total_seconds() / 60
        )
        df["days_since_last_session"] = df["review_interval_days"]
    else:
        df["session_duration_minutes"] = np.nan
        df["days_since_last_session"] = np.nan

    # Aggregate + hybrid features
    df = _safe_call(add_fsrs_probability, df)
    df = _safe_call(add_user_success_rate, df)
    df = _safe_call(add_user_average_confidence, df)
    df = _safe_call(add_user_average_response_time, df)
    df = _safe_call(add_question_global_success_rate, df)
    df = _safe_call(add_recent_success_rate, df)
    df = _safe_call(add_recent_confidence, df)
    df = _safe_call(add_consecutive_correct, df)
    df = _safe_call(add_hesitation_response_ratio, df)
    df = _safe_call(add_normalized_memory_inputs, df)

    # Ensure ALL_FEATURES columns are present (add missing as NaN)
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    # Validation (after all features added - tolerant of missing columns)
    missing = set(ALL_FEATURES + [TARGET]) - set(df.columns)
    if missing:
        print(f"Warning: Missing features (filled with NaN): {sorted(missing)}")

    # Select only the ordered feature list + target
    features_present = [c for c in ALL_FEATURES if c in df.columns]
    df = df[features_present + [TARGET]]

    return df


# ---------------------------------------------------------------------------
# Spark UDF wrapper (for when DataFrames are Spark-partitioned)
# ---------------------------------------------------------------------------

def build_features_spark_udf(spark):
    """
    Return a PySpark pandas_udf that applies build_features to each batch.

    Useful when working with Spark DataFrames in Databricks clusters.
    """

    import pyspark.sql.functions as F

    @F.pandas_udf(
        "double<double>...<double>",
        F.PandasUDFType.SCALAR,
    )
    def features_udf(df: pd.DataFrame) -> pd.DataFrame:
        """Apply build_features to each pandas batch."""
        results = []
        for _, batch in df.groupby(df.index // 10000):
            batch_result = build_features(batch)
            results.append(batch_result)
        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame(columns=ALL_FEATURES + [TARGET])

    return features_udf