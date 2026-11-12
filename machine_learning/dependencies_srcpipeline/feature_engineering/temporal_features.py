import pandas as pd


def add_review_interval(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["review_interval_days"] = (
        df["review_time"] - df["previous_review_time"]
    ).dt.total_seconds() / 86400

    return df


def add_hour_of_day(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["hour_of_day"] = df["review_time"].dt.hour

    return df


def add_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    # Day of week.
    # Monday = 0
    # Sunday = 6

    df = df.copy()

    df["day_of_week"] = df["review_time"].dt.dayofweek

    return df


def add_session_duration(df: pd.DataFrame) -> pd.DataFrame:
    #Session duration in minutes.

    df = df.copy()

    df["session_duration_minutes"] = (
        df["ended_at"] - df["started_at"]
    ).dt.total_seconds() / 60

    return df


def add_days_since_last_session(
    df: pd.DataFrame,
) -> pd.DataFrame:
 

    df = df.sort_values(["user_id", "started_at"]).copy()

    previous_session = df.groupby("user_id")["started_at"].shift(1)

    df["days_since_last_session"] = (
        df["started_at"] - previous_session
    ).dt.total_seconds() / 86400

    return df
