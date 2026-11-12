from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


# ---------------------------------------------------------------------------
# Schema definition for raw review data (adapted from existing ORM models)
# ---------------------------------------------------------------------------

RAW_REVIEW_SCHEMA = StructType(
    [
        StructField("user_id", StringType(), True),
        StructField("question_id", StringType(), True),
        StructField("correct", IntegerType(), True),
        StructField("confidence_score", DoubleType(), True),
        StructField("response_time_seconds", DoubleType(), True),
        StructField("hesitation_seconds", DoubleType(), True),
        StructField("question_difficulty", StringType(), True),
        StructField("question_position", IntegerType(), True),
        StructField("subject", StringType(), True),
        StructField("topic", StringType(), True),
        StructField("review_time", StringType(), True),  # ISO string or timestamp
        StructField("answer_changes", IntegerType(), True),
        # FSRS-derived (optional)
        StructField("recall_probability", DoubleType(), True),
    ]
)


# ---------------------------------------------------------------------------
# Helper: convert Spark DataFrame → pandas DataFrame (for sklearn compatibility)
# ---------------------------------------------------------------------------

def spark_to_pandas(spark_df: SparkDataFrame) -> pd.DataFrame:
    """Convert a PySpark DataFrame to pandas, caching the result."""
    return spark_df.toPandas()


# ---------------------------------------------------------------------------
# Stage 1: Column standardization & type casting
# ---------------------------------------------------------------------------

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names, strip whitespace, enforce expected names."""
    df = df.copy()
    df.columns = [
        c.strip().lower().replace(" ", "_") for c in df.columns
    ]
    return df


def cast_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to expected types matching the raw review schema."""
    df = df.copy()

    type_map = {
        "user_id": StringType(),
        "question_id": StringType(),
        "correct": IntegerType(),
        "confidence_score": DoubleType(),
        "response_time_seconds": DoubleType(),
        "hesitation_seconds": DoubleType(),
        "question_difficulty": StringType(),
        "question_position": IntegerType(),
        "subject": StringType(),
        "topic": StringType(),
        "review_time": StringType(),
        "answer_changes": IntegerType(),
        "recall_probability": DoubleType(),
    }

    for col_name, col_type in type_map.items():
        if col_name in df.columns:
            # Handle nullable / conversion errors gracefully
            try:
                if col_type == BooleanType():
                    df[col_name] = df[col_name].astype(bool)
                elif col_type == IntegerType():
                    df[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype("Int64")
                elif col_type == DoubleType():
                    df[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype("float64")
                elif col_type == StringType():
                    df[col_name] = df[col_name].astype(str)
            except (ValueError, TypeError):
                # If conversion fails, leave as-is; downstream handlers will deal with it
                pass

    return df


# ---------------------------------------------------------------------------
# Stage 2: Missing value handling
# ---------------------------------------------------------------------------

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Fill missing numeric columns; fill categorical with mode or constant."""
    df = df.copy()

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    for col in numeric_cols:
        if df[col].isna().any():
            if strategy == "median":
                fill = df[col].median()
            elif strategy == "mean":
                fill = df[col].mean()
            else:
                fill = fill_value
            df[col] = df[col].fillna(fill)

    for col in categorical_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna("unknown")

    return df


# ---------------------------------------------------------------------------
# Stage 3: Datetime parsing & feature extraction
# ---------------------------------------------------------------------------

def parse_review_time(df: pd.DataFrame, time_col: str = "review_time") -> pd.DataFrame:
    """Ensure review_time is datetime."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    return df


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour_of_day, day_of_week from review_time."""
    df = df.copy()
    if "review_time" in df.columns and pd.api.types.is_datetime64_any_dtype(df["review_time"]):
        df["hour_of_day"] = df["review_time"].dt.hour
        df["day_of_week"] = df["review_time"].dt.dayofweek  # 0=Monday
    else:
        df["hour_of_day"] = 0
        df["day_of_week"] = 0
    return df


# ---------------------------------------------------------------------------
# Stage 4: Difficulty mapping (mirrors existing feature_builder logic)
# ---------------------------------------------------------------------------

DIFFICULTY_SCORE_MAP = {
    "EASY": 1,
    "MEDIUM": 2,
    "HARD": 3,
}


def map_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """Map question_difficulty string to numeric score."""
    df = df.copy()
    if "question_difficulty" in df.columns:
        df["difficulty"] = df["question_difficulty"].map(DIFFICULTY_SCORE_MAP)
    else:
        df["difficulty"] = 0
    return df


# ---------------------------------------------------------------------------
# Stage 5: Persist transformed data
# ---------------------------------------------------------------------------

def write_delta(
    df: pd.DataFrame,
    path: Path,
    table_name: str,
    overwrite: bool = True,
) -> None:
    """Write a pandas DataFrame as a Delta table via Databricks."""

    # Convert to Spark DataFrame, write as Delta, then back to pandas if needed
    # In a real Databricks environment, use spark.write.format("delta")
    # Here we persist as CSV/Parquet as a stand-in for free-tier compatibility
    
    out_path = path / f"{table_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote transformed data to {out_path}")


# ---------------------------------------------------------------------------
# Public API: run the full transformation pipeline
# ---------------------------------------------------------------------------

def transform(
    input_path: str | Path,
    output_dir: str | Path,
    table_name: str = "reviews_transformed",
    spark=None,
) -> pd.DataFrame:
    """
    Full data transformation pipeline.

    Parameters
    ----------
    input_path : str | Path
        Path to raw CSV or parquet data.
    output_dir : str | Path
        Directory to persist transformed Delta/CSV data.
    table_name : str
        Name of the output table.
    spark : pyspark.SparkSession, optional
        Active Spark session (required for Spark-scale operations).

    Returns
    -------
    pd.DataFrame
        Cleaned, feature-ready pandas DataFrame.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    # Load data
    input_str = str(input_path) if not isinstance(input_path, str) else input_path
    if input_str.endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_parquet(input_str) if input_path else pd.DataFrame()

    print(f"Loaded {len(df)} rows from {input_path}")

    # Standardize
    df = standardize_columns(df)
    df = cast_column_types(df)

    # Parse timestamps
    df = parse_review_time(df)

    # Map difficulty
    df = map_difficulty(df)

    # Extract temporal features
    df = extract_temporal_features(df)

    # Handle missing values
    df = handle_missing_values(df, strategy="median")

    # Persist
    write_delta(df, path, table_name)

    print(f"Transformation complete: {len(df)} rows, {df.shape[1]} columns")
    return df