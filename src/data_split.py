import pandas as pd
from config import *
import numpy as np
from data_cleaning import clean_text_column


def sum_of_columns(dataframe, columns):
    temp = 0
    for col in columns:
        temp += dataframe[col]
    return temp


def data_prep(data, train=True):
    data = pd.read_csv(data)
    # -------------------------------------------------------------------------
    cols = DETECTION_CLASSES.copy()
    cols.remove('neutral')
    data['neutral'] = np.where(sum_of_columns(data, cols) > 0, 0, 1)
    # -------------------------------------------------------------------------
    data['comment_text'] = clean_text_column(data['comment_text'])
    print("Data Cleaned")

    x = data['comment_text'].values
    y = data[DETECTION_CLASSES].values

    if train:
        np.save("data/processed_data/x_train.npy", x)
        np.save("data/processed_data/y_train.npy", y)
    else:
        np.save("data/processed_data/x_test.npy", x)
        np.save("data/processed_data/y_test.npy", y)


if __name__ == '__main__':
    data_prep(TRAINING_DATA_LOCATION)
    data_prep(TEST_DATA_COMMENTS, False)
