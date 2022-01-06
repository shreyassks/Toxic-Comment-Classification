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


def tokenization(data, name="train"):

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(data)
    # -------------------------------------------------------------------------
    list_tokenized_train = tokenizer.texts_to_sequences(data)
    print('Data Tokenized to Sequences')
    # -------------------------------------------------------------------------
    word_index = tokenizer.word_index
    print(f'Found {len(word_index)} unique tokens')
    # -------------------------------------------------------------------------
    print('Saving tokens ...')
    with open(TOKENIZER_LOCATION, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # -------------------------------------------------------------------------
    x_tr = pad_sequences(list_tokenized_train, maxlen=max_seq_length, padding='post')
    np.save(f"data/processed_data/x_{name}_tokens.npy", x_tr)

    f = open("data/word_index.txt", "w")
    integer = len(word_index)
    f.write(str(integer))
    f.close()


def fast_text_embeddings():
    embeddings_index_fasttext = {}
    with open(EMBEDDING_FILE_LOCATION, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddings_index_fasttext[word] = np.asarray(values[1:], dtype='float32')

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
    np.save("data/embedding_matrix.npy", embedding_matrix_fasttext)


if __name__ == '__main__':
    x_train = np.load("data/processed_data/x_train.npy", allow_pickle=True)
    tokenization(x_train)
    fast_text_embeddings()


