from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from time import time

import joblib
import pandas as pd
import numpy as np
from scipy.stats import randint, uniform
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

import mlflow

from src_pipeline.features import build_features
from dependencies.training.preprocessing import build_preprocessor
from dependencies.utils.config import (
    FEATURE_DATA_DIR,
    MODEL_DIR,
    RANDOM_SEED,
    TEST_SIZE,
)

DATASET_PATH = FEATURE_DATA_DIR / "training_dataset.csv"

CV_FOLDS = 5

TUNE_ITER = 10

TUNE_FOLDS = 3

TUNE_SUBSAMPLE = 100_000

CALIBRATION_FOLDS = 3

# When --keep-incumbent is used, the previously serving calibrated model is
# preserved under this name so the eval gate can always compare candidate vs
# incumbent before the new one goes live.
INCUMBENT_NAME = "calibrated_best_incumbent"


# ---------------------------------------------------------------------------
# Model factories (adapted from src/training/trainer.py)
# ---------------------------------------------------------------------------

def _logistic_regression() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessor(scale_numerical=True),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    C=0.5,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def _random_forest(**overrides) -> Pipeline:
    config = {
        "n_estimators": 300,
        "min_samples_leaf": 5,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    }

    config.update(overrides)

    return Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessor(scale_numerical=False),
            ),
            (
                "model",
                RandomForestClassifier(**config),
            ),
        ]
    )


def _gradient_boosting(**overrides) -> Pipeline:
    config = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.9,
        "random_state": RANDOM_SEED,
    }

    config.update(overrides)

    return Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessor(scale_numerical=False),
            ),
            (
                "model",
                GradientBoostingClassifier(**config),
            ),
        ]
    )


def _hist_gradient_boosting(**overrides) -> Pipeline:
    config = {
        "max_iter": 400,
        "learning_rate": 0.06,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "random_state": RANDOM_SEED,
    }

    config.update(overrides)

    return Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessor(scale_numerical=False),
            ),
            (
                "model",
                HistGradientBoostingClassifier(**config),
            ),
        ]
    )


MODEL_FACTORIES = {
    "logistic_regression": _logistic_regression,
    "random_forest": _random_forest,
    "gradient_boosting": _gradient_boosting,
    "hist_gradient_boosting": _hist_gradient_boosting,
}


# ---------------------------------------------------------------------------
# Hyperparameter search (adapted from src/training/trainer.py)
# ---------------------------------------------------------------------------

PARAM_DISTRIBUTIONS = {
    "hist_gradient_boosting": {
        "model__learning_rate": uniform(0.03, 0.12),
        "model__max_iter": randint(250, 600),
        "model__max_leaf_nodes": randint(15, 63),
        "model__min_samples_leaf": randint(10, 60),
        "model__l2_regularization": uniform(0.01, 5.0),
    },
}


def evaluate_predictions(
    y_true,
    y_pred,
    y_proba,
) -> dict:
    return {
        "accuracy": round(
            float(accuracy_score(y_true, y_pred)),
            4,
        ),
        "precision": round(
            float(precision_score(y_true, y_pred)),
            4,
        ),
        "recall": round(
            float(recall_score(y_true, y_pred)),
            4,
        ),
        "f1": round(
            float(f1_score(y_true, y_pred)),
            4,
        ),
        "roc_auc": round(
            float(roc_auc_score(y_true, y_proba)),
            4,
        ),
        "log_loss": round(
            float(log_loss(y_true, y_proba)),
            4,
        ),
        "brier_score": round(
            float(brier_score_loss(y_true, y_proba)),
            4,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
        ).tolist(),
    }


def calibration_metrics(
    y_true,
    y_proba,
    n_bins: int = 10,
) -> dict:
    from sklearn.calibration import calibration_curve

    fraction_positive, mean_predicted = calibration_curve(
        y_true,
        y_proba,
        n_bins=n_bins,
        strategy="uniform",
    )

    return {
        "brier_score": round(
            float(brier_score_loss(y_true, y_proba)),
            4,
        ),
        "log_loss": round(
            float(log_loss(y_true, y_proba)),
            4,
        ),
        "calibration_curve": {
            "mean_predicted_probability": [
                round(float(x), 4) for x in mean_predicted
            ],
            "fraction_of_positives": [
                round(float(x), 4) for x in fraction_positive
            ],
        },
    }


def stratified_subsample(
    X,
    y,
    size: int,
):
    if len(X) <= size:
        return X, y

    frac = size / len(X)

    _, X_sub, _, y_sub = train_test_split(
        X,
        y,
        test_size=frac,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    return X_sub, y_sub


def tune_model(
    name: str,
    X_train,
    y_train,
) -> tuple:
    from dependencies.utils.config import RANDOM_SEED as _RS

    X_sub, y_sub = stratified_subsample(
        X_train,
        y_train,
        TUNE_SUBSAMPLE,
    )

    print(
        f"Tuning {name} on {len(X_sub)} rows "
        f"({TUNE_ITER} iter x {TUNE_FOLDS} folds)"
    )

    search = RandomizedSearchCV(
        estimator=MODEL_FACTORIES[name](),
        param_distributions=PARAM_DISTRIBUTIONS[name],
        n_iter=TUNE_ITER,
        cv=StratifiedKFold(
            n_splits=TUNE_FOLDS,
            shuffle=True,
            random_state=RANDOM_SEED,
        ),
        scoring="roc_auc",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        refit=False,
    )

    started = time()

    search.fit(X_sub, y_sub)

    print(
        f"Best CV AUC {search.best_score_:.4f} | "
        f"params {search.best_params_} | "
        f"{round(time() - started, 1)}s"
    )

    clean_params = {
        key.replace("model__", ""): value
        for key, value in search.best_params_.items()
    }

    return MODEL_FACTORIES[name](**clean_params), clean_params


# ---------------------------------------------------------------------------
# Decision threshold optimization
# ---------------------------------------------------------------------------

def optimize_threshold(
    pipeline,
    X_train,
    y_train,
    cv,
) -> float:
    print("Optimizing decision threshold on OOF predictions")

    oof_proba = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    thresholds = np.arange(0.05, 0.96, 0.01)

    accuracies = [
        ((oof_proba >= threshold).astype(bool) == y_train).mean()
        for threshold in thresholds
    ]

    best = float(thresholds[int(np.argmax(accuracies))])

    print(f"Best threshold {best:.2f}")

    return best


# ---------------------------------------------------------------------------
# Main training with MLflow integration
# ---------------------------------------------------------------------------

def run_training(
    dataset_path=None,
    tune: bool = False,
    keep_incumbent: bool = False,
) -> dict:
    print(
        f"Loading dataset: {dataset_path}"
    )

    # If dataset_path is already a DataFrame (pre-featured), use it directly
    if isinstance(dataset_path, pd.DataFrame):
        df = dataset_path
    else:
        df = load_dataset(dataset_path)

    print(
        f"Rows: {len(df)} | "
        f"Features: {df.shape[1]}"
    )
# TRAIN / TEST SPLIT


    X, y = _split_xy(df)
    print(
        f"Rows: {len(X)} | "
        f"Features: {X.shape[1]}"
    )


    # TRAIN / TEST SPLIT


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    print(
        f"Train: {len(X_train)} | "
        f"Test: {len(X_test)} | "
        f"Positive rate: {y.mean():.3f}"
    )


    # CROSS VALIDATION


    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    results = {}

    tuned_params = {}

    model_candidates = {}


    # MODEL EVALUATION


    with mlflow.start_run() as run:

        # Log basic params
        mlflow.log_param("dataset", str(dataset_path))
        mlflow.log_param("train_test_split", TEST_SIZE)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("cv_folds", CV_FOLDS)

        for name, factory in MODEL_FACTORIES.items():

            print(f"\n=== {name} ===")

            if tune and name in PARAM_DISTRIBUTIONS:

                pipeline, params = tune_model(
                    name,
                    X_train,
                    y_train,
                )

                tuned_params[name] = params

            else:

                pipeline = factory()

            model_candidates[name] = pipeline

            started = time()

            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
            )

            elapsed = round(
                time() - started,
                1,
            )

            print(
                f"CV ROC-AUC: "
                f"{scores.mean():.4f} "
                f"(+/- {scores.std():.4f}) "
                f"in {elapsed}s"
            )

            # Fit only for independent test evaluation.
            pipeline.fit(
                X_train,
                y_train,
            )

            proba = pipeline.predict_proba(
                X_test
            )[:, 1]

            pred = (
                proba >= 0.5
            ).astype(bool)

            metrics = evaluate_predictions(
                y_test,
                pred,
                proba,
            )

            metrics["cv_roc_auc_mean"] = round(
                float(scores.mean()),
                4,
            )

            metrics["cv_roc_auc_std"] = round(
                float(scores.std()),
                4,
            )

            metrics["cv_seconds"] = elapsed

            results[name] = metrics

            # Log model + metrics to MLflow
            mlflow.log_metric(f"cv_roc_auc_{name}", scores.mean())
            mlflow.log_metric(f"cv_roc_auc_std_{name}", scores.std())
            mlflow.log_metric(f"cv_seconds_{name}", elapsed)

            for key, val in metrics.items():
                if key not in ("cv_roc_auc_mean", "cv_roc_auc_std", "cv_seconds",
                               "confusion_matrix"):
                    mlflow.log_metric(f"test_{key}", val)

            mlflow.log_param(f"model_{name}", name)

            print(metrics)


    best_name = max(
        results,
        key=lambda name:
        results[name]["cv_roc_auc_mean"],
    )

    print(
        f"\nSelected base model: {best_name}"
    )

    print(
        "Selection criterion: "
        "highest mean 5-fold CV ROC-AUC"
    )

    print(
        f"CV ROC-AUC: "
        f"{results[best_name]['cv_roc_auc_mean']:.4f} "
        f"+/- "
        f"{results[best_name]['cv_roc_auc_std']:.4f}"
    )


    # --------------------------------------------------------
    # DECISION THRESHOLD
    # --------------------------------------------------------


    if best_name in tuned_params:

        best_pipeline, _ = tune_model(
            best_name,
            X_train,
            y_train,
        )

    else:

        best_pipeline = MODEL_FACTORIES[
            best_name
        ]()

    # --------------------------------------------------------
    # DECISION THRESHOLD
    # --------------------------------------------------------


    threshold = optimize_threshold(
        best_pipeline,
        X_train,
        y_train,
        cv,
    )


    # ISOTONIC CALIBRATION


    print(
        f"\nCalibrating {best_name} "
        f"using isotonic regression"
    )

    calibrated = CalibratedClassifierCV(
        estimator=best_pipeline,
        method="isotonic",
        cv=StratifiedKFold(
            n_splits=CALIBRATION_FOLDS,
            shuffle=True,
            random_state=RANDOM_SEED,
        ),
    )

    started = time()

    calibrated.fit(
        X_train,
        y_train,
    )

    calibration_seconds = round(
        time() - started,
        1,
    )


    # CALIBRATED TEST PROBABILITIES


    cal_proba = calibrated.predict_proba(
        X_test
    )[:, 1]

    cal_pred = (
        cal_proba >= threshold
    ).astype(bool)

    cal_metrics = evaluate_predictions(
        y_test,
        cal_pred,
        cal_proba,
    )

    cal_probability_metrics = calibration_metrics(
        y_test,
        cal_proba,
        n_bins=10,
    )

    # Add calibration metrics explicitly.
    cal_metrics["brier_score"] = (
        cal_probability_metrics["brier_score"]
    )

    cal_metrics["calibrated_log_loss"] = (
        cal_probability_metrics["log_loss"]
    )

    cal_metrics["calibration_curve"] = (
        cal_probability_metrics["calibration_curve"]
    )

    cal_metrics["calibration_seconds"] = (
        calibration_seconds
    )

    cal_metrics["base_model"] = best_name

    print(
        "\n=== CALIBRATED BEST MODEL ==="
    )

    print(
        f"Base model: {best_name}"
    )

    print(
        f"Accuracy: "
        f"{cal_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{cal_metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{cal_metrics['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{cal_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{cal_metrics['roc_auc']:.4f}"
    )

    print(
        f"Log Loss: "
        f"{cal_metrics['log_loss']:.4f}"
    )

    print(
        f"Brier Score: "
        f"{cal_metrics['brier_score']:.4f}"
    )

    print(
        f"Decision Threshold: "
        f"{threshold:.2f}"
    )


    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if keep_incumbent:

        incumbent_path = (
            MODEL_DIR
            / f"{INCUMBENT_NAME}.joblib"
        )

        calibrated_path = (
            MODEL_DIR
            / "calibrated_best.joblib"
        )

        if calibrated_path.exists() and not incumbent_path.exists():

            shutil.copy2(
                calibrated_path,
                incumbent_path,
            )

            print(
                f"Kept incumbent: {incumbent_path}"
            )

    # Save independently fitted base models.
    for name, pipeline in model_candidates.items():

        path = (
            MODEL_DIR
            / f"{name}.joblib"
        )

        joblib.dump(
            pipeline,
            path,
        )

        print(
            f"Saved {path}"
        )

    # Save calibrated best model.
    calibrated_path = (
        MODEL_DIR
        / "calibrated_best.joblib"
    )

    joblib.dump(
        calibrated,
        calibrated_path,
    )

    print(
        f"Saved {calibrated_path}"
    )


    # TRAINING REPORT


    report = {
        "generated_at": datetime.now(
            UTC
        ).isoformat(),

        "dataset": str(
            dataset_path
        ),

        "rows": len(df),

        "train_rows": len(
            X_train
        ),

        "test_rows": len(
            X_test
        ),

        "positive_rate": round(
            float(y.mean()),
            4,
        ),

        "model_selection": {
            "criterion": (
                "mean_5_fold_cv_roc_auc"
            ),
            "direction": "maximize",
            "selected_model": best_name,
        },

        "tuned": tuned_params,

        "decision_threshold": round(
            threshold,
            2,
        ),

        "models": results,

        "best_model": best_name,

        "calibrated_best": cal_metrics,
    }

    metrics_path = (
        MODEL_DIR
        / "training_report.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            report,
            handle,
            indent=2,
        )

    print(
        f"\nBest model: "
        f"{best_name} "
        f"@ threshold "
        f"{threshold:.2f}"
    )

    print(
        f"Report saved: "
        f"{metrics_path}"
    )

    # Log report artifact to MLflow
    mlflow.log_artifact(str(metrics_path))

    return report


def _load_dataset(
    path,
) -> pd.DataFrame:
    """Load dataset CSV, ensuring ALL_FEATURES + TARGET columns."""
    import pandas as pd

    from dependencies.feature_engineering.feature_manifest import ALL_FEATURES, TARGET

    df = pd.read_csv(path)

    missing = set(ALL_FEATURES + [TARGET]) - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    return df


def _split_xy(
    df: pd.DataFrame,
):
    from dependencies.feature_engineering.feature_manifest import ALL_FEATURES, TARGET

    return df[ALL_FEATURES].copy(), df[TARGET].astype(bool)


def main(
    tune: bool = False,
) -> None:

    run_training(
        tune=tune
    )


if __name__ == "__main__":
    main()