from sklearn.metrics import roc_auc_score
import numpy as np
from keras.models import load_model
from config import *
from data_preprocess import data_preprocess
import json


def evaluate(model, x_test, y_truth):
    predictions = model.predict(x_test)
    aucs = []
    for j in range(len(DETECTION_CLASSES)):
        auc = roc_auc_score(y_truth[:, j], predictions[:, j])
        aucs.append(auc)
    print(aucs)
    print(f'Average ROC_AUC Score: {np.mean(aucs)}')
    with open("scores.json", "w") as fd:
        json.dump({"auc": np.mean(aucs)}, fd, indent=4)


if __name__ == '__main__':
    lstm_model = load_model("models/toxicity_classifier.h5")
    x_t = np.load("data/processed_data/x_test.npy", allow_pickle=True)
    ind, x_te = data_preprocess(x_t)
    y_target = np.load("data/processed_data/y_test.npy", allow_pickle=True)
    evaluate(lstm_model, x_te, y_target)

