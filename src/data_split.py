from sklearn.model_selection import train_test_split
import pandas as pd
from config import *
import numpy as np
from data_cleaning import clean_text_column
import yaml


params = yaml.safe_load(open("params.yaml"))["data-split"]
split = params["split"]
seed = params["seed"]


def sum_of_columns(dataframe, columns):
    temp = 0
    for col in columns:
        temp += dataframe[col]
    return temp


def data_split(data):
    data = pd.read_csv(data)
    # -------------------------------------------------------------------------
    cols = DETECTION_CLASSES.copy()
    cols.remove('neutral')
    data['neutral'] = np.where(sum_of_columns(data, cols) > 0, 0, 1)
    # -------------------------------------------------------------------------
    data['comment_text'] = clean_text_column(data['comment_text'])
    print("Data Cleaned")

    train, test = train_test_split(data, test_size=split, random_state=seed)
    x_train = train['comment_text'].values
    y_train = train[DETECTION_CLASSES].values

    x_test = test['comment_text'].values
    y_test = test[DETECTION_CLASSES].values

    np.save("data/processed_data/x_train.npy", x_train)
    np.save("data/processed_data/x_test.npy", x_test)
    np.save("data/processed_data/y_train.npy", y_train)
    np.save("data/processed_data/y_test.npy", y_test)


if __name__ == '__main__':
    data_split(TRAINING_DATA_LOCATION)

