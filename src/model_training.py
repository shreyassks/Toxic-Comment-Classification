import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from keras.layers import Dense, Input, LSTM, Dropout
from keras.layers import GlobalMaxPool1D, Embedding
from keras.models import Model
from data_preprocess import tokenization
from config import *
import pickle
import yaml


params1 = yaml.safe_load(open("params.yaml"))["data-preprocess"]
vocab_size = params1["vocab_size"]
max_seq_length = params1["sequence_len"]
embedding_dim = params1["embedding_dim"]

params2 = yaml.safe_load(open("params.yaml"))["model-training"]
batch_size = params2["batch_size"]
epochs = params2["epochs"]
val_split = params2["val_split"]


def fast_text_embeddings():
    embeddings_index_fasttext = {}
    f = open(EMBEDDING_FILE_LOCATION, encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        embeddings_index_fasttext[word] = np.asarray(values[1:], dtype='float32')
    f.close()

    print(f'Total of {len(embeddings_index_fasttext)} word vectors are found.')

    with open(TOKENIZER_LOCATION, 'rb') as handle:
        tokenizer_model = pickle.load(handle)

    word_index = tokenizer_model.word_index
    embedding_matrix_fasttext = np.random.random((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index_fasttext.get(word)
        if embedding_vector is not None:
            embedding_matrix_fasttext[i] = embedding_vector
    print("Completed! Embedding Matrix")
    return embedding_matrix_fasttext


def build_lstm_model(data, target_classes, embeddings, token):
    inp = Input(shape=(max_seq_length,), dtype='int32')
    embedding_layer = Embedding(token+1,
                                embedding_dim, weights=[embeddings],
                                input_length=max_seq_length,
                                trainable=False,
                                name='fast_embeddings')
    embedded_sequences = embedding_layer(inp)
    x = LSTM(40, return_sequences=True, name='Lstm_layer')(embedded_sequences)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(30, activation="relu", kernel_initializer='he_uniform')(x)
    x = Dropout(0.1)(x)
    outs = Dense(7, activation="sigmoid", kernel_initializer='glorot_uniform')(x)
    # -------------------------------------------------------------------------
    model = Model(inputs=inp, outputs=outs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[tf.keras.metrics.AUC()])
    # -------------------------------------------------------------------------
    model.summary()
    # -------------------------------------------------------------------------
    history = model.fit(data,
                        target_classes,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=val_split,
                        verbose=1)
    model.save(MODEL_LOCATION)
    return model, history


def plot_training_history(history):
    # "Area under Curve"
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('AUC ROC Curve')
    plt.ylabel('AUC')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('auc.jpg')

    # "Log Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Log Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('log_loss.jpg')


def execute():
    x_train = np.load("data/processed_data/x_train.npy", allow_pickle=True)
    target_classes = np.load("data/processed_data/y_train.npy")

    tokenizer, x_tokens = tokenization(x_train)
    print('Saving tokens ...')
    with open(TOKENIZER_LOCATION, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    length = len(tokenizer.word_index)
    embeddings = fast_text_embeddings()
    lstm_model, history = build_lstm_model(x_tokens, target_classes, embeddings, length)
    plot_training_history(history)


# -------------------------------------------------------------------------
if __name__ == '__main__':
    execute()
