import numpy as np
from keras.models import load_model
from data_preprocess import data_preprocess
import json


def evaluate(model, x_test, y_truth):
    score = model.evaluate(x_test, y_truth, verbose=1)
    print("Test Score:", score[0])
    print("Test AUC:", score[1])
    with open("scores.json", "w") as fd:
        json.dump({"auc": score[1]}, fd, indent=4)


if __name__ == '__main__':
    lstm_model = load_model("models/toxicity_classifier.h5")
    x_t = np.load("data/processed_data/x_test.npy", allow_pickle=True)
    ind, x_te = data_preprocess(x_t)
    y_target = np.load("data/processed_data/y_test.npy", allow_pickle=True)
    evaluate(lstm_model, x_te, y_target)

