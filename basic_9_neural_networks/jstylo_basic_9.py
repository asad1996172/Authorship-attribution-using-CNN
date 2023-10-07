import nltk
import numpy as np
import pyphen
from keras.preprocessing import text

dic = pyphen.Pyphen(lang="nl_NL")


def total_words(input_text: str) -> int:
    """
    Calculate the total number of words in the input text.

    :param input_text: The input text to count the words from.
    :return: The total number of words in the input text.
    """
    words = text.text_to_word_sequence(
        input_text, filters=",.?!\"'`;:-()&$", lower=True, split=" "
    )
    return len(words)


def text_complexity(input_text: str) -> float:
    """
    Calculate the complexity of the input text as the ratio of unique words to total words.

    :param input_text: The input text to calculate the complexity of.
    :return: The complexity of the input text as a float value.
    """
    words = text.text_to_word_sequence(
        input_text, filters=",.?!\"'`;:-()&$", lower=True, split=" "
    )
    unique_word_count = len(set(words))
    return unique_word_count / len(words)


def sentences_count(input_text: str) -> int:
    """
    Calculate the total number of sentences in the input text.

    :param input_text: The input text to count the sentences from.
    :return: The total number of sentences in the input text.
    """
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    sentences = tokenizer.tokenize(input_text)
    return len(sentences)


def average_sentence_length(input_text: str) -> float:
    """
    Calculate the average sentence length in the input text.

    :param input_text: A string containing the input text.
    :return: A float representing the average sentence length in the input text.
    """
    return total_words(input_text) / sentences_count(input_text)


def average_syllables_in_word(input_text: str) -> float:
    """
    Calculate the average number of syllables in words for the input text.

    :param input_text: The input text to calculate the average number of syllables in words.
    :return: The average number of syllables in words for the input text.
    """
    words = text.text_to_word_sequence(
        input_text, filters=",.?!\"'`;:-()&$", lower=True, split=" "
    )
    syllable_counts = [len(dic.inserted(word).split("-")) for word in words]
    return np.mean(syllable_counts)


def total_complex_words(input_text: str) -> int:
    """
    Calculate the total number of complex words (with 3 or more syllables) in the input text.

    :param input_text: The input text to analyze.
    :return: The total number of complex words in the input text.
    """
    words = text.text_to_word_sequence(
        input_text, filters=",.?!\"'`;:-()&$", lower=True, split=" "
    )
    return sum(1 for word in words if len(dic.inserted(word).split("-")) >= 3)


def gunning_fog_readability_index(input_text: str) -> float:
    """
    Calculate the Gunning Fog readability index for the input text.
    
    :param input_text: A string representing the input text.
    :return: A float representing the Gunning Fog readability index for the input text.
    """
    return 0.4 * (
        average_sentence_length(input_text)
        + 100 * (total_complex_words(input_text) / total_words(input_text))
    )


def characters_count(input_text: str) -> int:
    """
    Calculates the total number of characters in the input text.

    :param input_text: The input text to count the characters from.
    :return: The total number of characters in the input text.
    """
    return len(input_text)


def total_number_of_letters(input_text: str) -> int:
    """
    Calculate the total number of letters in the input text.

    :param input_text: The input text to count the letters from.
    :return: The total number of letters in the input text.
    """
    words = text.text_to_word_sequence(
        input_text, filters=",.?!\"'`;:-()&$", lower=False, split=" "
    )
    return sum(letter.isalpha() for word in words for letter in word)


def total_syllables(input_text: str) -> int:
    """
    Calculate the total number of syllables in the input text.

    :param input_text: The input text to calculate the syllables from.
    :return: The total number of syllables in the input text.
    """
    words = text.text_to_word_sequence(
        input_text, filters=",.?!\"'`;:-()&$", lower=True, split=" "
    )
    return sum(len(dic.inserted(word).split("-")) for word in words)


def flesch_reading_ease_score(input_text: str) -> float:
    """
    Calculate the Flesch Reading Ease score for the input text.

    :param input_text: The text to calculate the score for.
    :return: The Flesch Reading Ease score for the input text.
    """
    return (
        206.835
        - 1.015 * (total_words(input_text) / sentences_count(input_text))
        - 84.6 * (total_syllables(input_text) / total_words(input_text))
    )


def calculate_features(input_text: str) -> list:
    """
    Calculate a list of various textual features for the input text.

    :param input_text: The input text for which features are to be calculated.
    :return: A list of various textual features for the input text.
    """
    return [
        total_words(input_text),
        text_complexity(input_text),
        sentences_count(input_text),
        average_sentence_length(input_text),
        average_syllables_in_word(input_text),
        gunning_fog_readability_index(input_text),
        characters_count(input_text),
        total_number_of_letters(input_text),
        flesch_reading_ease_score(input_text),
    ]
