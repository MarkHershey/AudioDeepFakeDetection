import json
import os
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skm
from scipy.interpolate import interp1d
from scipy.optimize import brentq


def compute_roc_auc_eer(y_true, y_pred) -> Tuple[float, float]:
    # Compute the ROC curve and the AUC
    fpr, tpr, thresholds = skm.roc_curve(y_true, y_pred, pos_label=1)
    roc_auc = skm.auc(fpr, tpr)
    # Compute the Equal Error Rate (EER)
    fnr = 1 - tpr
    # Ref: https://stackoverflow.com/a/46026962
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return roc_auc, eer


def alt_compute_eer(y_true, y_pred) -> float:
    # Ref: https://github.com/scikit-learn/scikit-learn/issues/15247#issuecomment-542138349
    fpr, tpr, thresholds = skm.roc_curve(y_true, y_pred, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer


def main():
    np.random.seed(0)

    # Load the data
    y_true = np.random.randint(2, size=500)
    y_pred = np.random.randint(2, size=500)

    # Compute the ROC curve and the AUC
    fpr, tpr, thresholds = skm.roc_curve(y_true, y_pred, pos_label=1)
    roc_auc = skm.auc(fpr, tpr)
    # Compute the Equal Error Rate (EER)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    print(f"EER: {eer:.3f}")

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([1, 0], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.show()


def compute_metrics_for_file(
    filename: Union[str, Path]
) -> Tuple[float, float, float, float, float]:
    filepath: Path = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} does not exist.")
    with filepath.open("r") as f:
        data = json.load(f)

    acc: float = skm.accuracy_score(data["y_true"], data["y_pred"])
    f1: float = skm.f1_score(data["y_true"], data["y_pred"])
    eer: float = alt_compute_eer(data["y_true"], data["y_pred"])
    roc_auc, eer2 = compute_roc_auc_eer(data["y_true"], data["y_pred"])

    return acc, f1, roc_auc, eer, eer2


def compute_all():
    save_dir = Path(__file__).parent / "saved"
    export_filepath = save_dir / "RESULT.md"
    result = {}

    for exp_name in os.listdir(save_dir):
        pred_filepath = save_dir / exp_name / "best_pred.json"
        if not pred_filepath.is_file():
            continue

        acc, f1, roc_auc, eer, eer2 = compute_metrics_for_file(pred_filepath)
        result[exp_name] = dict(
            acc=f"{acc:.3f}",
            f1=f"{f1:.3f}",
            roc_auc=f"{roc_auc:.4f}",
            eer=f"{eer:.4f}",
            eer2=f"{eer2:.4f}",
        )

    to_write = [
        "# Empirical Results",
        " ",
        "- Accuracy",
        "- F1 score",
        "- Area Under the Receiver Operating Characteristic Curve (ROC AUC)",
        "- Equal Error Rate (EER)",
        " ",
        "| Experiment | Accuracy | F1 | ROC AUC | EER | EER2 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for exp_name in sorted(result.keys(), key=lambda x: result[x]["f1"]):
        d = result[exp_name]
        to_write.append(
            f"| {exp_name} | {d['acc']} | {d['f1']} | {d['roc_auc']} | {d['eer']} | {d['eer2']} |"
        )

    with export_filepath.open("w") as f:
        f.write("\n".join(to_write))


if __name__ == "__main__":
    compute_all()
