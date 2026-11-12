from __future__ import annotations

import json
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from dependencies.utils.config import MODEL_DIR, RANDOM_SEED


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(model_name: str = "calibrated_best") -> Any:
    """
    Load a trained model from the model directory.

    Parameters
    ----------
    model_name : str
        Name of the model file (without .joblib extension).

    Returns
    -------
    Trained sklearn Pipeline or CalibratedClassifierCV object.
    """
    model_path = MODEL_DIR / f"{model_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")
    return model


# ---------------------------------------------------------------------------
# Feature schema validation for inference
# ---------------------------------------------------------------------------

# Import feature manifest for schema validation
from dependencies.feature_engineering.feature_manifest import ALL_FEATURES, TARGET

# Import the preprocessor builder from training module
from dependencies.training.preprocessing import build_preprocessor


# ---------------------------------------------------------------------------
# Batch inference: score a DataFrame / Spark DataFrame
# ---------------------------------------------------------------------------

def predict_proba(
    model,
    df: pd.DataFrame,
    return_std: bool = False,
) -> dict:
    """
    Compute prediction probabilities for a feature-complete DataFrame.

    Parameters
    ----------
    model : sklearn Pipeline
        Trained model pipeline (must have .predict_proba method).
    df : pd.DataFrame
        Feature data with ALL_FEATURES columns.
    return_std : bool
        If True and model supports it, return prediction std (not applicable
        for most classifiers; kept for interface compatibility).

    Returns
    -------
    dict with keys:
        - "proba": np.ndarray of shape (n_samples, 2) or (n_samples,)
        - "pred": np.ndarray of bool predictions
        - "threshold": float threshold used
    """
    # Ensure all required features are present
    missing = set(ALL_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required features: {missing}. "
            f"Expected columns: {ALL_FEATURES}"
        )

    # Align columns to match training order
    df = df[[c for c in ALL_FEATURES if c in df.columns]].copy()

    proba = model.predict_proba(df)[:, 1]

    # Use 0.5 threshold by default; in production this would be stored with model
    threshold = 0.5
    pred = (proba >= threshold).astype(bool)

    result = {
        "proba": proba,
        "pred": pred,
        "threshold": threshold,
    }

    if return_std:
        # Most classifiers don't provide std; return confidence margin instead
        result["confidence_margin"] = np.abs(proba - 0.5)

    return result


def predict_batch(
    model,
    df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Run batch inference and return a DataFrame with predictions attached.

    Parameters
    ----------
    model : sklearn Pipeline
        Trained model.
    df : pd.DataFrame
        Feature data with ALL_FEATURES columns.
    threshold : float
        Decision threshold for binary classification.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 'prediction' and 'probability' columns added.
    """
    result = predict_proba(model, df)

    df = df.copy()
    df["prediction"] = result["pred"]
    df["probability"] = result["proba"]
    df["confidence_margin"] = np.abs(result["proba"] - threshold)

    return df


# ---------------------------------------------------------------------------
# Single-record inference (for REST API / widget use)
# ---------------------------------------------------------------------------

def predict_single(
    model,
    record: Dict[str, float],
    threshold: float = 0.5,
) -> Dict[str, any]:
    """
    Predict for a single record dict.

    Parameters
    ----------
    model : sklearn Pipeline
        Trained model pipeline.
    record : Dict[str, float]
        Feature name → value mapping for one review instance.
    threshold : float
        Decision threshold.

    Returns
    -------
    Dict with keys: prediction, probability, confidence_margin
    """
    import pandas as pd

    df = pd.DataFrame([record])
    result = predict_proba(model, df, threshold=threshold)

    return {
        "prediction": int(result["pred"][0]),
        "probability": float(result["proba"][0]),
        "confidence_margin": float(result["confidence_margin"][0])
        if "confidence_margin" in result
        else None,
    }


# ---------------------------------------------------------------------------
# Model artifact persistence (free-tier compatible: local only)
# ---------------------------------------------------------------------------

def save_model(
    model: Any,
    path: str | Path = None,
    name: str = "calibrated_best",
) -> Path:
    """
    Save a trained model to disk (local filesystem only on free tier).

    Parameters
    ----------
    model : sklearn object
        Trained pipeline to persist.
    path : str | Path | None
        Directory to save in; defaults to MODEL_DIR.
    name : str
        Model filename (without .joblib).

    Returns
    -------
    Path where the model was saved.
    """
    save_dir = Path(path) if path else MODEL_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{name}.joblib"
    joblib.dump(model, save_path)

    print(f"Model saved to {save_path}")
    return save_path


def list_models(
    directory: str | Path = None,
) -> List[str]:
    """
    List saved model joblib files in a directory.

    Parameters
    ----------
    directory : str | Path | None
        Directory to scan; defaults to MODEL_DIR.

    Returns
    -------
    List of model names (filenames without extension).
    """
    scan_dir = Path(directory) if directory else MODEL_DIR
    if not scan_dir.exists():
        return []

    models = [
        f.stem
        for f in scan_dir.iterdir()
        if f.suffix == ".joblib"
    ]
    return [p.stem for p in models]


# ---------------------------------------------------------------------------
# MLflow model registration (free tier: local tracking only)
# ---------------------------------------------------------------------------

def log_model_to_mlflow(
    model,
    artifact_path: str = "model",
    registered_model_name: str | None = None,
) -> None:
    """
    Log a model to MLflow.

    On Databricks free tier, this uses local artifact root. For production,
    set MLFLOW_TRACKING_URI to a remote server (paid tier).

    Parameters
    ----------
    model : sklearn Pipeline
        Trained model to log.
    artifact_path : str
        Key under which the model artifact is stored.
    registered_model_name : str | None
        Name to register the model under; skip on free tier.
    """
    import mlflow

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )

        run_id = run.info.run_id
        print(
            f"Model logged to MLflow run {run_id} "
            f"as artifact '{artifact_path}'"
        )