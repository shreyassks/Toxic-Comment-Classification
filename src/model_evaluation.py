import numpy as np
from config import DETECTION_CLASSES
from keras.models import load_model
from data_preprocess import tokenization
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json


def evaluate(model, x, y):
    score = model.evaluate(x, y, verbose=1)
    print("Test Score:", score[0])
    print("Test AUC:", score[1])
    with open("scores.json", "w") as fd:
        json.dump({"auc": score[1]}, fd, indent=4)


def confusion_plots(model, x):
    predictions = model.predict(x)
    y_pred = predictions.copy()
    y_pred[:] = y_pred[:] > 0.5
    y_pred = y_pred.astype(int)

    cm = multilabel_confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))

    f, axes = plt.subplots(2, 3, figsize=(25, 15))
    axes = axes.ravel()
    for i in range(6):
        disp = ConfusionMatrixDisplay(confusion_matrix(y_test[:, i],
                                                       y_pred[:, i]),
                                      display_labels=[0, i])
        disp.plot(ax=axes[i], values_format='.0f')
        disp.ax_.set_title(f'{DETECTION_CLASSES[i]}')
        disp.ax_.set_xlabel('')
        disp.ax_.set_ylabel('')
        disp.im_.colorbar.remove()

    plt.subplots_adjust(wspace=0.10, hspace=0.1)
    f.colorbar(disp.im_, ax=axes)
    plt.savefig("confusion_matrix.jpg")
    print("Saved Confusion Matrix plot")
    return y_pred


def auc_roc_plots(y_target, y_preds):
    n_classes = 7
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_target[:, i], y_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_target.ravel(), y_preds.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    fig = plt.figure(figsize=(8, 6))
    colors = ['darkorange', 'red', 'black', 'yellow', 'blue', 'green']
    for i in range(6):
        plt.plot(fpr[i], tpr[i], color=colors[i],
                 label=f'ROC curve -> {DETECTION_CLASSES[i]}, area=({round(roc_auc[i], 2)})')

    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 11}, loc='lower right')
    plt.savefig("auc_roc.jpg")


if __name__ == '__main__':
    lstm_model = load_model("models/toxicity_classifier.h5")
    x_test = np.load("data/processed_data/x_test.npy", allow_pickle=True)
    y_test = np.load("data/processed_data/y_test.npy", allow_pickle=True)

    _, x_tokens = tokenization(x_test, True)
    evaluate(lstm_model, x_tokens, y_test)
    preds = confusion_plots(lstm_model, x_tokens)
    auc_roc_plots(y_test, preds)
    print("Completed")

