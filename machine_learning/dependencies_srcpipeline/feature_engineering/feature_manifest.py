


# Numerical Features

NUMERICAL_FEATURES = [
    # Student history
    "total_revisions",
    "success_rate",
    "average_confidence",
    "average_response_time",
    "average_hesitation",
    # Question information
    "difficulty",
    "word_count",
    "character_count",
    # Session information
    "session_duration_minutes",
    "question_position_in_session",
    "days_since_last_session",
    # Review history
    "review_interval_days",
    "repetition_number",

    "last_review_confidence_score",

    "last_review_response_time",

    "last_review_hesitation",
    # Behaviour
    "answer_changes",
    # Hybrid + aggregate features
    "fsrs_recall_probability",
    "user_success_rate",
    "user_average_confidence",
    "user_average_response_time",
    "question_global_success_rate",
    "recent_success_rate_5",
    "recent_confidence_5",
    "consecutive_correct",
    "hesitation_response_ratio",
    "normalized_interval_days",
    "normalized_repetition_number",
    "normalized_avg_response_time",
    "normalized_avg_hesitation",
]


# Categorical Features

CATEGORICAL_FEATURES = [
    "subject",
    "topic",
    "hour_of_day",
    "day_of_week",
]


# Boolean Features

BOOLEAN_FEATURES = [
    "last_review_correct",
    "had_fsrs_estimate",
]


# Target

TARGET = "correct"


# Complete Feature List

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
