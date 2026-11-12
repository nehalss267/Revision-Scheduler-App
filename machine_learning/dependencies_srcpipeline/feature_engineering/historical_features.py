
import numpy as np
import pandas as pd


def sort_reviews(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(by=["user_id", "question_id", "review_time"]).copy()


GROUP_COLUMNS = ["user_id", "question_id"]


def running_average(
    df: pd.DataFrame,
    column: str,
    ) -> pd.Series:

    grouped = grouped_reviews(df)

    previous_sum = (
        grouped[column]
        .transform(lambda x: x.shift().cumsum())
        .astype(float)
    )

    previous_count = _safe_previous_count(grouped)

    return previous_sum / previous_count


def _safe_previous_count(grouped):


    return grouped.cumcount().replace(0, np.nan)


def grouped_reviews(df: pd.DataFrame):


    return df.groupby(GROUP_COLUMNS)


def add_total_revisions(df: pd.DataFrame) -> pd.DataFrame:

    df["total_revisions"] = grouped_reviews(df).cumcount()

    return df


def add_previous_correct(df: pd.DataFrame) -> pd.DataFrame:

    df["last_review_correct"] = (
        grouped_reviews(df)["correct"]
        .shift(1)
    )


    return df


def add_previous_confidence_score(df: pd.DataFrame) -> pd.DataFrame:


    df["last_review_confidence_score"] = (
        grouped_reviews(df)["confidence_score"]
        .shift(1)
    )

    return df


def add_previous_response_time(df: pd.DataFrame) -> pd.DataFrame:


    df["last_review_response_time"] = df.groupby(["user_id", "question_id"])[
        "response_time_seconds"
    ].shift(1)

    return df


def add_previous_hesitation(df: pd.DataFrame) -> pd.DataFrame:



    df["last_review_hesitation"] = df.groupby(["user_id", "question_id"])[
        "hesitation_seconds"
    ].shift(1)

    return df


def add_success_rate(df: pd.DataFrame) -> pd.DataFrame:



    grouped = grouped_reviews(df)

    previous_correct = (
        grouped["correct"]
        .transform(lambda x: x.shift().cumsum())
        .astype(float)
    )

    previous_reviews = _safe_previous_count(grouped)

    df["success_rate"] = previous_correct / previous_reviews

    return df


def add_average_confidence(df: pd.DataFrame) -> pd.DataFrame:


    df["average_confidence"] = running_average(
        df,
        "confidence_score",
    )

    return df


def add_average_response_time(df: pd.DataFrame) -> pd.DataFrame:



    df["average_response_time"] = running_average(
        df,
        "response_time_seconds",
    )

    return df


def add_average_hesitation(df: pd.DataFrame) -> pd.DataFrame:


    df["average_hesitation"] = running_average(
        df,
        "hesitation_seconds",
    )

    return df
