"""
This script classifies text based on the 
provided dataset and number of authors.

Usage:
    python3 classifier.py \
        --number_of_authors <NUM_AUTHORS>\
        --dataset <DATASET_TYPE> \
        --train_test_data_path <PATH_TO_DATA> \
        --glove_model_path <PATH_TO_GLOVE_MODEL>

Sample run command:
    python3 classifier.py \
        --number_of_authors 5 \
        --dataset blogs \
        --train_test_data_path ../train_test_data \
        --glove_model_path ../../glove.840B.300d.zip
"""
import argparse
import os
import pickle
import urllib.request
import zipfile
from typing import Dict, List

import numpy as np
from colorama import Fore
from keras.layers import Conv1D, Dense, Dropout, Embedding, Flatten, MaxPooling1D, Merge
from keras.models import Sequential
from keras.preprocessing import text
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.utils.visualize_util import plot


def download_glove(filepath: str):
    """
    Download the GloVe model.

    :param filepath: Path to save the model file.
    """
    print("Downloading GloVe model. This might take a while...")
    url = "http://nlp.stanford.edu/data/glove.840B.300d.zip"
    urllib.request.urlretrieve(url, filepath)


def load_model(filepath: str) -> dict:
    """
    Load the GloVe model.

    :param filepath: Path to the model file.
    """
    parent_dir = os.path.dirname(filepath)
    print(f"Loading GloVe model from {filepath}...")

    # Check if the model exists. If not, download it.
    if not os.path.exists(filepath):
        download_glove(filepath)
        # Extract the zip file
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(parent_dir)

    embeddings_index = {}
    with open(filepath.replace(".zip", ".txt"), "r", encoding="utf-8") as file:
        for line in file:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype="float32")
                if len(coefs) == 300:
                    embeddings_index[word] = coefs
            except:
                # skipping words that cause errors
                continue

    print(f"Found {len(embeddings_index)} word vectors.")
    return embeddings_index


def model_cnn_word_word(
    embedding_matrix: Dict[str, np.ndarray],
    word_index: Dict[str, int],
    embedding_dim: int,
    max_sequence_length: int,
    number_of_authors: int,
) -> Sequential:
    """
    Define a CNN model with two parallel embedding layers (trainable and non-trainable).

    :param embedding_matrix (Dict[str, np.ndarray]): A dictionary of word embeddings.
    :param word_index (Dict[str, int]): A dictionary of word indices.
    :param embedding_dim (int): The dimension of the word embeddings.
    :param max_sequence_length (int): Maximum length of the input sequence.
    :return: A compiled CNN model.
    """

    embed1 = Sequential()
    embed1.add(
        Embedding(
            len(word_index) + 1,
            embedding_dim,
            weights=[embedding_matrix],
            input_length=max_sequence_length,
            trainable=False,
        )
    )
    embed2 = Sequential()
    embed2.add(
        Embedding(
            len(word_index) + 1,
            embedding_dim,
            weights=[embedding_matrix],
            input_length=max_sequence_length,
            trainable=True,
        )
    )

    model = Sequential()
    model.add(Merge([embed1, embed2], mode="concat", concat_axis=-1))

    model.add(Conv1D(64, 5, activation="relu"))
    model.add(MaxPooling1D(max_sequence_length - 5 + 1))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_authors, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])
    plot(model, to_file='model.png')

    print(model.summary())

    return model


def fill_in_missing_words_with_zeros(
    embeddings_index: Dict[str, np.ndarray],
    word_index: Dict[str, int],
    EMBEDDING_DIM: int,
) -> np.ndarray:
    """
    This function takes in a dictionary of word embeddings and a dictionary of word indices,
    and returns a matrix of word embeddings where any missing words are filled in with zeros.


    :param embeddings_index (Dict[str, np.ndarray]): A dictionary of word embeddings.
    :param word_index (Dict[str, int]): A dictionary of word indices.
    :param EMBEDDING_DIM (int): The dimension of the word embeddings.
    :return: A matrix of word embeddings where any missing words are filled in with zeros.
    """
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def create_tokenizer(lines: List[str]) -> Tokenizer:
    """
    Create a tokenizer object and fit it on the given lines of text.

    :param lines: A list of strings representing the text to fit the tokenizer on.
    :return: A tokenizer object that has been fit on the given text.
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def load_pickle_data(path: str):
    """
    Loads data from a pickle file.

    :param path: Path to the pickle file.
    :return: Data loaded from the pickle file.
    """
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data


def get_clean_text(input_text: str) -> str:
    """
    Cleans the provided text by removing special characters and
    converting the text to a sequence of words.

    :param input_text: Raw text string to be cleaned.
    :return: Cleaned text string.
    """
    clean_text_list = text.text_to_word_sequence(
        input_text, filters="", lower=False, split=" "
    )
    clean_text = " ".join(clean_text_list)
    return clean_text


def prepare_data_for_classification(data: List[tuple]) -> tuple:
    """
    Each row contains data in the following format: (input_text, author_label)
    This function extracts the features for inputText and
    authorLabel from each row.

    :param data: List of tuples containing the raw text and the author id.
    :return: List of tuples containing the cleaned text and the author id.
    """
    X_data = []
    y_data = []
    for input_text, author_label in data:
        clean_text = get_clean_text(input_text)
        X_data.append(clean_text)
        y_data.append(author_label)
    return X_data, y_data


def train_classifier(
    X_train: List[List[float]],
    y_train: List[int],
    glove_filepath: str,
    number_of_authors: int,
    embedding_dim: int = 300,
) -> tuple:
    """
    Trains a classifier using the provided training data.

    :param number_of_authors: Number of authors in the dataset.
    :param X_train: List of feature vectors for training data.
    :param y_train: List of author ids for training data.
    :return: Trained classifier and StandardScaler used to scale the data.
    """
    tokenizer = create_tokenizer(X_train)
    sequences = tokenizer.texts_to_sequences(X_train)
    max_sequence_length = max([len(article.split()) for article in X_train])
    vocab_size = len(tokenizer.word_index) + 1

    print(f"\nMax document length: {max_sequence_length}")
    print(f"Vocabulary size: {vocab_size}\n")

    x_train = pad_sequences(sequences, maxlen=max_sequence_length)
    y_train = to_categorical(y_train)

    print(f"Shape of data tensor: {x_train.shape}")
    print(f"Shape of label tensor: {y_train.shape}\n")

    glove_embedding_matrix = {}
    print("Filling non existing words\n")
    glove_embedding_matrix = fill_in_missing_words_with_zeros(
        glove_embedding_matrix, tokenizer.word_index, embedding_dim
    )

    model = model_cnn_word_word(
        glove_embedding_matrix,
        tokenizer.word_index,
        embedding_dim,
        max_sequence_length,
        number_of_authors,
    )
    model.fit([x_train, x_train], y_train, nb_epoch=15, batch_size=50)

    return tokenizer, model, max_sequence_length


def test_classifier(
    X_test: List[List[float]],
    y_test: List[int],
    tokenizer: Tokenizer,
    max_sequence_length: int,
    model: Sequential,
) -> None:
    """
    Tests the classifier using the provided test data.

    :param X_test: List of feature vectors for test data.
    :param y_test: List of author ids for test data.
    :param tokenizer: Tokenizer used to tokenize the data.
    :param max_sequence_length: Maximum length of the input sequence.
    :param model: Trained classifier.
    """
    sequences = tokenizer.texts_to_sequences(X_test)
    y_test = to_categorical(y_test)
    x_test = pad_sequences(sequences, maxlen=max_sequence_length)
    _, acc = model.evaluate([x_test, x_test], y_test, verbose=0)

    print("Test Accuracy: %f" % (acc * 100))


def main(
    number_of_authors: int,
    dataset_type: str,
    train_test_data_path: str,
    glove_model_path: str,
) -> None:
    """
    Main function to process the dataset, train a classifier, and test it.

    :param number_of_authors: Number of authors in the dataset.
    :param dataset_type: Type of dataset (e.g., 'amt', 'blogs', etc.).
    :param train_test_data_path: Directory path
        containing training and testing pickle data files.
    :param glove_model_path: Path to the GloVe model.
    """

    print(Fore.GREEN)
    print(f"\nNumber of Authors: {number_of_authors}")
    print(f"Dataset Type: {dataset_type}")
    print(f"Train/Test Data Path: {train_test_data_path}\n")

    train_data_path = (
        f"{train_test_data_path}/{str(number_of_authors)}"
        f"_train_files_{dataset_type}.pkl"
    )
    test_data_path = (
        f"{train_test_data_path}/{str(number_of_authors)}"
        f"_test_files_{dataset_type}.pkl"
    )

    train_data = load_pickle_data(train_data_path)
    test_data = load_pickle_data(test_data_path)

    X_train, y_train = prepare_data_for_classification(train_data)
    X_test, y_test = prepare_data_for_classification(test_data)

    print(f"Total Train Articles: {len(train_data)}")
    print(f"Total Test Articles: {len(test_data)}\n")

    tokenizer, model, max_sequence_length = train_classifier(
        X_train, y_train, glove_model_path, number_of_authors
    )
    test_classifier(X_test, y_test, tokenizer, max_sequence_length, model)
    print(Fore.RESET)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test CNN classifier.")

    parser.add_argument(
        "--number_of_authors",
        type=int,
        help="Number of authors to keep for classification.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["blogs"],
        help='Choose "blogs" dataset.',
    )
    parser.add_argument(
        "--train_test_data_path", type=str, help="Path to the train/test data."
    )
    parser.add_argument(
        "--glove_model_path", type=str, help="Path of glove.840B.300d.zip model."
    )

    args = parser.parse_args()

    main(
        args.number_of_authors,
        args.dataset,
        args.train_test_data_path,
        args.glove_model_path,
    )
