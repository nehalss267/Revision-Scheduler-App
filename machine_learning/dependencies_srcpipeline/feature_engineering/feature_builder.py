import pandas as pd

from src.feature_engineering.aggregate_features import (
    add_aggregate_features,
)
from src.feature_engineering.cleaning import (
    convert_datetime_columns,
)
from src.feature_engineering.historical_features import (
    add_average_confidence,
    add_average_hesitation,
    add_average_response_time,
    add_previous_confidence_score,
    add_previous_correct,
    add_previous_hesitation,
    add_previous_response_time,
    add_success_rate,
    add_total_revisions,
    sort_reviews,
)
from src.feature_engineering.temporal_features import (
    add_day_of_week,
    add_days_since_last_session,
    add_hour_of_day,
    add_review_interval,
    add_session_duration,
)
from src.feature_engineering.validators import validate_features


def build_features(df: pd.DataFrame) -> pd.DataFrame:

    DIFFICULTY_SCORE_MAP = {
        "EASY": 1,
        "MEDIUM": 2,
        "HARD": 3,
    }

    df["difficulty"] = df["question_difficulty"].map(DIFFICULTY_SCORE_MAP)

    df["question_position_in_session"] = df["question_position"]

    df = convert_datetime_columns(df)

    # Sort reviews
    df = sort_reviews(df)

    # Historical features

    df = add_total_revisions(df)

    df = add_previous_correct(df)

    df = add_previous_confidence_score(df)

    df = add_previous_response_time(df)

    df = add_previous_hesitation(df)

    df = add_success_rate(df)

    df = add_average_confidence(df)

    df = add_average_response_time(df)

    df = add_average_hesitation(df)

    # Temporal features

    df = add_review_interval(df)

    df = add_hour_of_day(df)

    df = add_day_of_week(df)

    df = add_session_duration(df)

    df = add_days_since_last_session(df)

    # Aggregate + hybrid features

    df = add_aggregate_features(df)

    # Validation

    validate_features(df)

    return df
