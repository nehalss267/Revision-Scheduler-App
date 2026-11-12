

from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulator.config import SIMULATION_LIMITS


def add_fsrs_probability(df: pd.DataFrame) -> pd.DataFrame:


    df["fsrs_recall_probability"] = df["recall_probability"]

    df["had_fsrs_estimate"] = (
        df["recall_probability"].notna().astype(int)
    )

    return df.drop(columns=["recall_probability"])


def _expanding_mean_shifted(
    df: pd.DataFrame,
    group_columns: list[str],
    sort_columns: list[str],
    column: str,
) -> pd.Series:


    ordered = df.sort_values(sort_columns, kind="mergesort")

    values = ordered.groupby(group_columns, sort=False)[
        column
    ].transform(lambda s: s.shift().expanding().mean())

    return pd.Series(values.values, index=ordered.index)


def add_user_success_rate(df: pd.DataFrame) -> pd.DataFrame:
    df["user_success_rate"] = _expanding_mean_shifted(
        df,
        ["user_id"],
        ["user_id", "review_time"],
        "correct",
    )

    return df


def add_user_average_confidence(df: pd.DataFrame) -> pd.DataFrame:
    df["user_average_confidence"] = _expanding_mean_shifted(
        df,
        ["user_id"],
        ["user_id", "review_time"],
        "confidence_score",
    )

    return df


def add_user_average_response_time(df: pd.DataFrame) -> pd.DataFrame:
    df["user_average_response_time"] = _expanding_mean_shifted(
        df,
        ["user_id"],
        ["user_id", "review_time"],
        "response_time_seconds",
    )

    return df


def add_question_global_success_rate(df: pd.DataFrame) -> pd.DataFrame:


    df["question_global_success_rate"] = _expanding_mean_shifted(
        df,
        ["question_id"],
        ["review_time", "question_id"],
        "correct",
    )

    return df


def _rolling_mean_shifted(
    series: pd.Series,
    window: int,
) -> pd.Series:
    return series.shift().rolling(window, min_periods=1).mean()


def add_recent_success_rate(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(
        ["user_id", "question_id"],
        sort=False,
    )["correct"]

    df["recent_success_rate_5"] = grouped.transform(
        lambda s: _rolling_mean_shifted(s, 5)
    )

    return df


def add_recent_confidence(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(
        ["user_id", "question_id"],
        sort=False,
    )["confidence_score"]

    df["recent_confidence_5"] = grouped.transform(
        lambda s: _rolling_mean_shifted(s, 5)
    )

    return df


def _streak_from_previous(series: pd.Series) -> pd.Series:


    previous = series.shift()

    is_one = previous == 1

    blocks = is_one.ne(is_one.shift()).cumsum()

    streak = previous.groupby(blocks).cumcount() + 1

    streak = streak.where(is_one, 0.0)

    return streak.where(previous.notna(), np.nan)


def add_consecutive_correct(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(
        ["user_id", "question_id"],
        sort=False,
    )["correct"]

    df["consecutive_correct"] = grouped.transform(_streak_from_previous)

    return df


def add_hesitation_response_ratio(df: pd.DataFrame) -> pd.DataFrame:
    denominator = df["average_response_time"].replace(0, np.nan)

    df["hesitation_response_ratio"] = (
        df["average_hesitation"] / denominator
    )

    return df


def add_normalized_memory_inputs(df: pd.DataFrame) -> pd.DataFrame:

    df["normalized_interval_days"] = (
        df["review_interval_days"]
        / SIMULATION_LIMITS.max_review_interval_days
    ).clip(0, 1)

    df["normalized_repetition_number"] = (
        df["repetition_number"]
        / SIMULATION_LIMITS.max_repetition_number
    ).clip(0, 1)

    df["normalized_avg_response_time"] = (
        df["average_response_time"]
        / SIMULATION_LIMITS.max_response_time
    ).clip(0, 1)

    df["normalized_avg_hesitation"] = (
        df["average_hesitation"]
        / SIMULATION_LIMITS.max_hesitation
    ).clip(0, 1)

    return df


def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:

    df = add_fsrs_probability(df)

    df = add_user_success_rate(df)

    df = add_user_average_confidence(df)

    df = add_user_average_response_time(df)

    df = add_question_global_success_rate(df)

    df = add_recent_success_rate(df)

    df = add_recent_confidence(df)

    df = add_consecutive_correct(df)

    df = add_hesitation_response_ratio(df)

    df = add_normalized_memory_inputs(df)

    return df
