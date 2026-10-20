"""Draw a side-by-side comparison of the latest MLflow run of each model."""
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split

from src import data
from src.features import FEATURES
from src.train import CANDIDATES, ROOT, SEED

OUT = ROOT / "screenshots" / "model_comparison.png"
METRICS = ["cv_accuracy", "cv_precision", "cv_recall", "cv_roc_auc"]
COLORS = {"logistic_regression": "#2a9d8f", "random_forest": "#e9c46a",
          "gradient_boosting": "#e76f51"}


def latest_runs(client: MlflowClient, experiment_id: str) -> dict:
    runs = client.search_runs([experiment_id], order_by=["attributes.start_time DESC"],
                              filter_string="attributes.status = 'FINISHED'")
    found = {}
    for r in runs:
        name = r.data.tags.get("mlflow.runName")
        if name in CANDIDATES and name not in found:
            found[name] = r
    return found


def main(out: Path = OUT):
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    client = MlflowClient()
    exp = client.get_experiment_by_name("heart-disease")
    runs = latest_runs(client, exp.experiment_id)

    df = data.load_clean()
    _, X_test, _, y_test = train_test_split(
        df[FEATURES], df["target"], test_size=0.2, stratify=df["target"], random_state=SEED)

    fig, (ax_roc, ax_bar) = plt.subplots(1, 2, figsize=(13, 5))
    for name, run in runs.items():
        model = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc, name=name,
                                       curve_kwargs={"color": COLORS[name]})
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set_title("ROC curves on the hold-out set")

    width = 0.8 / len(runs)
    x = np.arange(len(METRICS))
    for i, (name, run) in enumerate(runs.items()):
        vals = [run.data.metrics[m] for m in METRICS]
        ax_bar.bar(x + i * width, vals, width, label=name, color=COLORS[name])
    ax_bar.set_xticks(x + width * (len(runs) - 1) / 2)
    ax_bar.set_xticklabels([m.replace("cv_", "") for m in METRICS])
    ax_bar.set_ylim(0.6, 1.02)
    ax_bar.set_title("5-fold cross-validated metrics (training split)")
    ax_bar.legend(loc="upper center", ncol=3, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)

    mlflow.set_experiment("heart-disease")
    with mlflow.start_run(run_name="model_comparison"):
        mlflow.log_artifact(str(out), artifact_path="plots")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
