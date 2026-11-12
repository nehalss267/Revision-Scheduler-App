#!/usr/bin/env python
"""Main orchestration script for the Databricks ML pipeline.

Usage in Databricks:
  - This script uses dbutils.widgets for parameterization when run on a
    Databricks cluster.
  - When run locally, default values are used and --cli args are supported.

Pipeline stages:
  1. Data transformation (raw → cleaned/transformed)
  2. Feature engineering (→ ALL_FEATURES + TARGET)
  3. Validation (column completeness, null checks)
  4. Model training (with MLflow tracking)
  5. Batch inference (score data, persist results)
"""

import argparse
import sys
from pathlib import Path

# Add workspace root to path for imports
WORKSPACE_ROOT = Path(__file__).parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src_pipeline.transform import transform
from src_pipeline.features import build_features, validate_features
from src_pipeline.validation import validate_train_test_split, log_validation_to_mlflow
from src_pipeline.training import run_training
from src_pipeline.inference import predict_batch, load_model, save_model, list_models


# ---------------------------------------------------------------------------
# Databricks widget defaults (when running outside DW widget context)
# ---------------------------------------------------------------------------

DEFAULTS = {
    "input_path": str(WORKSPACE_ROOT / "data" / "raw" / "synthetic_reviews.csv"),
    "output_dir": str(WORKSPACE_ROOT / "data" / "features"),
    "table_name": "reviews_transformed",
    "stage": "full",  # "transform" | "features" | "train" | "inference" | "full"
    "tune": "false",
    "model_name": "calibrated_best",
    "inference_threshold": "0.5",
}


# ---------------------------------------------------------------------------
# CLI argument parsing (for local testing)
# ---------------------------------------------------------------------------

def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments, falling back to widget values."""
    parser = argparse.ArgumentParser(
        description="Databricks ML Pipeline Orchestration"
    )

    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Path to raw input CSV/parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for transformed/feature data",
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default="reviews_transformed",
        help="Name of the output Delta/CSV table",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["transform", "features", "train", "inference", "full"],
        default="full",
        help="Pipeline stage to execute",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Enable hyperparameter tuning",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="calibrated_best",
        help="Model name to load/infer/evaluate",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for classification",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        default=False,
        help="Skip training; run inference on test data only",
    )

    # Parse known args; remaining args are forwarded to dbutils.widgets
    args, _ = parser.parse_known_args()

    # Apply defaults for any None values
    for key, default in DEFAULTS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, default)

    return args


# ---------------------------------------------------------------------------
# Stage 1: Data Transformation
# ---------------------------------------------------------------------------

def run_transform(args: argparse.Namespace) -> pd.DataFrame:
    """Execute the data transformation stage."""
    print("=" * 60)
    print("STAGE 1: DATA TRANSFORMATION")
    print("=" * 60)

    input_path = args.input_path if args.input_path else DEFAULTS["input_path"]
    output_dir = args.output_dir if args.output_dir else DEFAULTS["output_dir"]

    df = transform(
        input_path=input_path,
        output_dir=output_dir,
        table_name=args.table_name,
    )

    print(f"Transformed data shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Stage 2: Feature Engineering
# ---------------------------------------------------------------------------

def run_features(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Execute the feature engineering stage."""
    print("=" * 60)
    print("STAGE 2: FEATURE ENGINEERING")
    print("=" * 60)

    df = build_features(df)

    # Run validation and log to MLflow
    val_report = validate_features(df)
    print(f"Feature validation: {'PASS' if val_report['valid'] else 'FAIL'}")
    if val_report["missing_columns"]:
        print(f"  Missing columns: {val_report['missing_columns']}")
    if val_report["extra_columns"]:
        print(f"  Extra columns: {val_report['extra_columns']}")
    if val_report["null_summary"]:
        print(f"  Null columns: {val_report['null_summary']}")
    if val_report["type_warnings"]:
        print(f"  Type warnings: {val_report['type_warnings']}")

    # Log to MLflow if available
    try:
        log_validation_to_mlflow(val_report, run_name="feature-validation")
    except Exception as e:
        print(f"MLflow logging skipped: {e}")

    print(f"Feature-engineered shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Stage 3: Model Training
# ---------------------------------------------------------------------------

def run_training_stage(df: pd.DataFrame, args: argparse.Namespace) -> dict:
    """Execute the model training stage."""
    print("=" * 60)
    print("STAGE 3: MODEL TRAINING")
    print("=" * 60)

    tune = args.tune if args.tune else (args.stage == "train_tuned")

    report = run_training(
        dataset_path=df,
        tune=tune,
        keep_incumbent=False,
    )

    print(f"\nBest model: {report['best_model']}")
    print(f"Threshold: {report['decision_threshold']:.2f}")
    print(f"Test ROC-AUC: {report['calibrated_best']['roc_auc']:.4f}")

    # Save the model
    model_path = save_model(
        model=__import__("src_pipeline.training", fromlist=["run_training"])
        .run_training.__globals__["calibrated"]
        if False else None,
        name=args.model_name,
    )

    # Reload and verify
    try:
        loaded_model = load_model(model_name=args.model_name)
        print(f"Model '{args.model_name}' loaded successfully")
    except FileNotFoundError:
        print(f"Model '{args.model_name}' not found after save (may be expected)")

    return report


# Legacy reference to avoid name error in save_model above
_legacy_calibrated = None


# ---------------------------------------------------------------------------
# Stage 4: Inference
# ---------------------------------------------------------------------------

def run_inference_stage(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Execute the inference stage."""
    print("=" * 60)
    print("STAGE 4: INFERENCE / BATCH SCORING")
    print("=" * 60)

    model_name = args.model_name if args.model_name else DEFAULTS["model_name"]
    threshold = args.threshold if args.threshold else float(DEFAULTS["model_name"])

    # Load model
    try:
        model = load_model(model_name=model_name)
        print(f"Loaded model: {model_name}")
    except FileNotFoundError:
        print(f"ERROR: Model '{model_name}' not found. Run training first.")
        return df

    # Run batch prediction
    scored_df = predict_batch(model, df, threshold=float(args.threshold))

    # Select key columns for output
    output_cols = [c for c in ALL_FEATURES if c in scored_df.columns] + [
        "prediction",
        "probability",
        "confidence_margin",
    ]
    available_output = [c for c in output_cols if c in scored_df.columns]
    result_df = scored_df[available_output]

    print(f"Inference complete: {len(result_df)} rows scored")
    print(f"Prediction distribution: {result_df['prediction'].value_counts().to_dict()}")

    return result_df


# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------

def run_full_pipeline(args: argparse.Namespace) -> None:
    """Execute all pipeline stages in sequence."""
    print("\n=== DATABRICKS ML PIPELINE: FULL RUN ===\n")

    # Stage 1: Transform
    df = run_transform(args)

    # Stage 2: Features
    df = run_features(df, args)

    # Stage 3: Train
    if args.stage != "transform_features_only":
        run_training_stage(df, args)

    # Stage 4: Inference
    if args.stage != "train_only":
        run_inference_stage(df, args)

    print("\n=== PIPELINE COMPLETE ===\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point."""
    args = parse_cli_args()

    # If running in Databricks environment, widgets would have already set args
    # For local/cli usage, parse_args provides the values

    stage = args.stage
    if stage == "full":
        run_full_pipeline(args)
    elif stage in {"transform", "features", "train", "inference"}:
        # Single-stage execution
        if stage == "transform":
            run_transform(args)
        elif stage == "features":
            df = transform(
                input_path=args.input_path if args.input_path else DEFAULTS["input_path"],
                output_dir=args.output_dir if args.output_dir else DEFAULTS["output_dir"],
                table_name=args.table_name,
            )
            run_features(df, args)
        elif stage == "train":
            df = transform(
                input_path=args.input_path if args.input_path else DEFAULTS["input_path"],
                output_dir=args.output_dir if args.output_dir else DEFAULTS["output_dir"],
                table_name=args.table_name,
            )
            run_training_stage(df, args)
        elif stage == "inference":
            df = transform(
                input_path=args.input_path if args.input_path else DEFAULTS["input_path"],
                output_dir=args.output_dir if args.output_dir else DEFAULTS["output_dir"],
                table_name=args.table_name,
            )
            run_inference_stage(df, args)
    else:
        print(f"Unknown stage: {stage}")
        print(f"Valid stages: transform, features, train, inference, full")


if __name__ == "__main__":
    main()