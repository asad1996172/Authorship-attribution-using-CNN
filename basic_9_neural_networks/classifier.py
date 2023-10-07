"""
This script classifies text based on the 
provided dataset and number of authors.

Usage:
    python3 classifier.py \
        --number_of_authors <NUM_AUTHORS>\
        --dataset <DATASET_TYPE> \
        --train_test_data_path <PATH_TO_DATA>

Sample run command:
    python3 classifier.py \
        --number_of_authors 5 \
        --dataset blogs \
        --train_test_data_path ../train_test_data
"""

import argparse
import pickle
from typing import List

from colorama import Fore
from jstylo_basic_9 import calculate_features
from keras.preprocessing import text
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


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
        features = calculate_features(clean_text)
        X_data.append(features)
        y_data.append(author_label)
    return X_data, y_data


def train_classifier(
    number_of_authors: int, X_train: List[List[float]], y_train: List[int]
) -> tuple:
    """
    Trains a classifier using the provided training data.

    :param number_of_authors: Number of authors in the dataset.
    :param X_train: List of feature vectors for training data.
    :param y_train: List of author ids for training data.
    :return: Trained classifier and StandardScaler used to scale the data.
    """
    if number_of_authors == 5:
        clf = MLPClassifier(
            hidden_layer_sizes=(5, 7),
            activation="relu",
            learning_rate_init=0.009,
            random_state=100,
        )
    else:
        clf = MLPClassifier(
            hidden_layer_sizes=(5, 9),
            activation="relu",
            learning_rate_init=0.009,
            random_state=100,
        )

    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)

    clf.fit(X_train, y_train)

    print("Train Accuracy : ", clf.score(X_train, y_train))
    return clf, scalar


def test_classifier(
    X_test: List[List[float]],
    y_test: List[int],
    clf: MLPClassifier,
    scalar: StandardScaler,
) -> None:
    """
    Tests the classifier using the provided test data.

    :param X_test: List of feature vectors for test data.
    :param y_test: List of author ids for test data.
    :param clf: Trained classifier.
    :param scalar: StandardScaler used to scale the data.
    """
    X_test = scalar.transform(X_test)
    y_pred = clf.predict(X_test)

    print("Test Accuracy : ", accuracy_score(y_test, y_pred))


def main(number_of_authors: int, dataset_type: str, train_test_data_path: str) -> None:
    """
    Main function to process the dataset, train a classifier, and test it.

    :param number_of_authors: Number of authors in the dataset.
    :param dataset_type: Type of dataset (e.g., 'amt', 'blogs', etc.).
    :param train_test_data_path: Directory path
        containing training and testing pickle data files.
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

    clf, scalar = train_classifier(number_of_authors, X_train, y_train)
    test_classifier(X_test, y_test, clf, scalar)
    print(Fore.RESET)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and test neural networks using basic 9 feature set."
    )

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

    args = parser.parse_args()

    main(args.number_of_authors, args.dataset, args.train_test_data_path)
