import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skm


def compute_roc_auc_eer(y_true, y_pred):
    # Compute the ROC curve and the AUC
    fpr, tpr, thresholds = skm.roc_curve(y_true, y_pred, pos_label=1)
    roc_auc = skm.auc(fpr, tpr)
    # Compute the Equal Error Rate (EER)
    fnr = 1 - tpr
    # Ref: https://stackoverflow.com/a/46026962
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return roc_auc, eer


def alt_compute_eer(y_true, y_pred):
    # Ref: https://github.com/scikit-learn/scikit-learn/issues/15247#issuecomment-542138349
    from scipy.interpolate import interp1d
    from scipy.optimize import brentq

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


if __name__ == "__main__":
    main()
