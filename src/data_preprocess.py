import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from config import *
import yaml
import nltk
nltk.download('omw-1.4')


params = yaml.safe_load(open("params.yaml"))["data-preprocess"]
vocab_size = params["vocab_size"]
max_seq_length = params["sequence_len"]
embedding_dim = params["embedding_dim"]


def tokenization(data, token=False):

    if token:
        with open(TOKENIZER_LOCATION, 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(data)

    # -------------------------------------------------------------------------
    list_tokenized = tokenizer.texts_to_sequences(data)
    print('Data Tokenized to Sequences')
    # -------------------------------------------------------------------------
    x_tr = pad_sequences(list_tokenized, maxlen=max_seq_length, padding='post')

    return tokenizer, x_tr


# if __name__ == '__main__':
#     x_train = np.load("data/processed_data/x_train.npy", allow_pickle=True)
#     tokenization(x_train)

