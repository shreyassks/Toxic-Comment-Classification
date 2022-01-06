import numpy as np
from keras.models import load_model
from data_preprocess import tokenization
import json


def evaluate(model, x, y):
    score = model.evaluate(x, y, verbose=1, batch_size=32)
    print("Test Score:", score[0])
    print("Test AUC:", score[1])
    with open("scores.json", "w") as fd:
        json.dump({"auc": score[1]}, fd, indent=4)


if __name__ == '__main__':
    lstm_model = load_model("models/toxicity_classifier.h5")
    x_te = np.load("data/processed_data/x_test.npy", allow_pickle=True)
    tokenization(x_te, name="test")
    x_test = np.load("data/processed_data/x_test_tokens.npy", allow_pickle=True)
    y_test = np.load("data/processed_data/y_test.npy", allow_pickle=True)
    evaluate(lstm_model, x_test, y_test)

