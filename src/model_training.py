import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Dense, Input, LSTM, Dropout
from keras.layers import GlobalMaxPool1D, Embedding
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from config import *
from data_preprocess import data_preprocess
import yaml


params1 = yaml.safe_load(open("params.yaml"))["data-preprocess"]
vocab_size = params1["vocab_size"]
max_seq_length = params1["sequence_len"]
embedding_dim = params1["embedding_dim"]

params2 = yaml.safe_load(open("params.yaml"))["model-training"]
batch_size = params2["batch_size"]
epochs = params2["epochs"]
val_split = params2["val_split"]
# -------------------------------------------------------------------------


def build_lstm_model(data, target_classes, embeddings, word_index):
    inp = Input(shape=(max_seq_length,), dtype='int32')
    embedded_sequences = Embedding(word_index+1, embedding_dim, weights=[embeddings],
                                   input_length=max_seq_length,
                                   trainable=False,
                                   name='embeddings')(inp)
    x = LSTM(40, return_sequences=True, name='lstm_layer')(embedded_sequences)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(30, activation="relu", kernel_initializer='he_uniform')(x)
    x = Dropout(0.1)(x)
    preds = Dense(7, activation="sigmoid", kernel_initializer='glorot_uniform')(x)
    # -------------------------------------------------------------------------
    model = Model(inputs=inp, outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # -------------------------------------------------------------------------
    model.summary()
    # -------------------------------------------------------------------------
    checkpoint = ModelCheckpoint(filepath=MODEL_LOCATION,  # saves the 'best' model
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=True)
    # -------------------------------------------------------------------------
    history = model.fit(data,
                        target_classes,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=val_split,
                        verbose=1)
    # -------------------------------------------------------------------------
    model.save(MODEL_LOCATION)
    # -------------------------------------------------------------------------
    return model, history


# -------------------------------------------------------------------------
def plot_training_history(history):
    # "Accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('plots/accuracy.jpg')

    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('plots/loss.jpg')


def execute():
    x_train = np.load("data/processed_data/x_train.npy", allow_pickle=True)
    ind, data = data_preprocess(x_train)
    target_classes = np.load("data/processed_data/y_train.npy")
    embeddings = np.load("data/embedding_matrix.npy")

    lstm_model, history = build_lstm_model(data, target_classes, embeddings, ind)
    plot_training_history(history)


# -------------------------------------------------------------------------
if __name__ == '__main__':
    execute()
