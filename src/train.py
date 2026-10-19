"""Train candidate models, tune them with CV, track runs in MLflow, save the best."""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)

from src import data
from src.features import FEATURES, build_pipeline

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
SCORING = ["accuracy", "precision", "recall", "roc_auc"]

# Hyperparameter grids (keys are prefixed with the pipeline step name).
CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000, random_state=SEED),
        {"model__C": [0.01, 0.1, 1, 10], "model__penalty": ["l2"]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=SEED),
        {"model__n_estimators": [100, 300],
         "model__max_depth": [3, 5, None],
         "model__min_samples_leaf": [1, 3]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=SEED),
        {"model__n_estimators": [100, 200],
         "model__learning_rate": [0.03, 0.1],
         "model__max_depth": [2, 3]},
    ),
}


def evaluate_holdout(pipe, X_test, y_test) -> dict:
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "test_accuracy": accuracy_score(y_test, pred),
        "test_precision": precision_score(y_test, pred),
        "test_recall": recall_score(y_test, pred),
        "test_roc_auc": roc_auc_score(y_test, proba),
    }


def save_plots(pipe, X_test, y_test, name: str, out: Path) -> list:
    out.mkdir(parents=True, exist_ok=True)
    cm_path, roc_path = out / f"{name}_confusion.png", out / f"{name}_roc.png"
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, ax=ax, cmap="Blues")
    ax.set_title(f"{name}: confusion matrix")
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax)
    ax.set_title(f"{name}: ROC curve")
    fig.savefig(roc_path, bbox_inches="tight")
    plt.close(fig)
    return [cm_path, roc_path]


def main():
    df = data.load_clean() if data.CLEAN_PATH.exists() else data.clean(data.load_raw())
    X, y = df[FEATURES], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("heart-disease")
    plot_dir = ROOT / "screenshots" / "model_plots"

    results = {}
    for name, (estimator, grid) in CANDIDATES.items():
        with mlflow.start_run(run_name=name):
            search = GridSearchCV(build_pipeline(estimator), grid, cv=cv,
                                  scoring="roc_auc", n_jobs=-1)
            search.fit(X_train, y_train)
            best = search.best_estimator_

            # Cross-validated metrics of the tuned pipeline on the training split.
            cvres = cross_validate(best, X_train, y_train, cv=cv, scoring=SCORING)
            cv_metrics = {f"cv_{m}": cvres[f"test_{m}"].mean() for m in SCORING}
            cv_metrics.update({f"cv_{m}_std": cvres[f"test_{m}"].std() for m in SCORING})
            test_metrics = evaluate_holdout(best, X_test, y_test)

            mlflow.log_params({k.replace("model__", ""): v
                               for k, v in search.best_params_.items()})
            mlflow.log_param("model_type", name)
            mlflow.log_metrics({**cv_metrics, **test_metrics})
            for p in save_plots(best, X_test, y_test, name, plot_dir):
                mlflow.log_artifact(str(p), artifact_path="plots")
            mlflow.sklearn.log_model(best, name="model", serialization_format="cloudpickle")

            results[name] = {"pipeline": best, **cv_metrics, **test_metrics,
                             "params": search.best_params_}
            print(f"{name:22s} cv_auc={cv_metrics['cv_roc_auc']:.3f} "
                  f"cv_recall={cv_metrics['cv_recall']:.3f} "
                  f"test_auc={test_metrics['test_roc_auc']:.3f}")

    # Select by CV ROC-AUC, breaking near-ties with CV recall (missing a sick patient is costly).
    winner = max(results, key=lambda n: (round(results[n]["cv_roc_auc"], 2),
                                         results[n]["cv_recall"]))
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(results[winner]["pipeline"], MODEL_DIR / "model.joblib")
    summary = {n: {k: float(v) for k, v in r.items() if k not in ("pipeline", "params")}
               for n, r in results.items()}
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps({"selected": winner, "models": summary}, indent=2))
    print(f"\nSelected model: {winner} -> {MODEL_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()
