import pandas as pd

from src.feature_engineering.feature_manifest import (
    ALL_FEATURES,
    TARGET,
)


def validate_features(df: pd.DataFrame):


    missing = set(ALL_FEATURES + [TARGET]) - set(df.columns)

    if missing:
        raise ValueError(f"Missing features: {missing}")
    
    if "confidence_score" in df.columns:
        invalid = df[
            (df["confidence_score"] < 0)
            | (df["confidence_score"] > 1)
        ]

        if not invalid.empty:
            raise ValueError(
                "confidence_score must be between 0 and 1."
            )

    if "average_confidence" in df.columns:
        invalid = df[
            (df["average_confidence"] < 0)
            | (df["average_confidence"] > 1)
        ]

        if not invalid.empty:
            raise ValueError(
                "average_confidence must be between 0 and 1."
            )

    NON_NEGATIVE_COLUMNS = [
        "response_time_seconds",
        "hesitation_seconds",
        "average_response_time",
        "average_hesitation",
        "review_interval_days",
        "session_duration_minutes",
    ]

    for column in NON_NEGATIVE_COLUMNS:

        if column not in df.columns:
            continue

        if (df[column] < 0).any():
            raise ValueError(
                f"{column} contains negative values."
            )

    if "answer_changes" in df.columns and (
        df["answer_changes"] < 0
    ).any():
        raise ValueError(
            "answer_changes cannot be negative."
        )

    if "success_rate" in df.columns:

        invalid = df[
            (df["success_rate"] < 0)
            | (df["success_rate"] > 1)
        ]

        if not invalid.empty:
            raise ValueError(
                "success_rate must be between 0 and 1."
            )
    return True
