import os
import re
from typing import List

import nltk
import numpy as np
import spacy
from keras.preprocessing import text
from sortedcontainers import SortedDict

nlp = spacy.load("en_core_web_sm")
cur_dir_path = os.path.dirname(os.path.realpath(__file__))


def get_clean_text(input_text: str) -> str:
    """
    Convert input text to a clean string of words.

    :param input_text: A string of text to clean.
    :return: A string of clean text.
    """
    clean_text: List[str] = text.text_to_word_sequence(
        input_text, filters="", lower=False, split=" "
    )

    clean_text: str = "".join(str(e) + " " for e in clean_text)
    return clean_text


def characters_count(input_text: str) -> int:
    """
    Calculates character count including spaces.

    :param input_text: A string of text to count characters.
    :return An integer representing the count of characters.
    """
    input_text = input_text.lower()
    char_count = len(str(input_text))
    return char_count


def average_character_per_word(input_text: str) -> float:
    """
    Calculates the average number of characters per word.

    :param input_text: A string of text to calculate the average number of characters per word.
    :return: A float representing the average number of characters per word.
    """
    words: List[str] = text.text_to_word_sequence(
        input_text,
        filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"',
        lower=True,
        split=" ",
    )
    input_text: str = input_text.lower().replace(" ", "")
    char_count: int = len(str(input_text))

    avg_char_count: float = char_count / len(words)
    return avg_char_count


def frequency_of_letters(input_text: str) -> List[float]:
    """
    Calculates the frequency of letters in the input text.

    :param input_text: A string of text to calculate the frequency of letters.
    :return: A list of floats representing the frequency of each letter in the input text.
    """
    input_text = str(input_text).lower()  # because it's case sensitive
    input_text = input_text.lower().replace(" ", "")
    characters = "abcdefghijklmnopqrstuvwxyz"
    chars_frequency_dict = {}
    for c in range(0, len(characters)):
        char = characters[c]
        chars_frequency_dict[char] = 0
        for i in str(input_text):
            if char == i:
                chars_frequency_dict[char] = chars_frequency_dict[char] + 1

    # making a vector out of the frequencies
    vector_of_frequencies = [0] * len(characters)
    total_count = sum(list(chars_frequency_dict.values()))
    for c in range(0, len(characters)):
        char = characters[c]
        vector_of_frequencies[c] = chars_frequency_dict[char] / total_count

    return vector_of_frequencies


def most_common_letter_bigrams(input_text: str) -> List[float]:
    """
    Calculates the frequency of most common letter bigrams in the input text.

    :param input_text: A string of text to calculate the frequency 
        of most common letter bigrams.
    :return A list of floats representing the frequency of each 
        most common letter bigram in the input text.
    """
    bigrams = [
        "th",
        "he",
        "in",
        "er",
        "an",
        "re",
        "nd",
        "at",
        "on",
        "nt",
        "ha",
        "es",
        "st",
        "en",
        "ed",
        "to",
        "it",
        "ou",
        "ea",
        "hi",
        "is",
        "or",
        "ti",
        "as",
        "te",
        "et",
        "ng",
        "of",
        "al",
        "de",
        "se",
        "le",
        "sa",
        "si",
        "ar",
        "ve",
        "ra",
        "ld",
        "ur",
    ]

    bigrams_dict = {}
    for bigram in bigrams:
        bigrams_dict[bigram] = True

    bigram_counts = {}
    for bigram in bigrams_dict:
        bigram_counts[bigram] = 0

    total_count = 0
    words = text.text_to_word_sequence(
        input_text, filters=",.?!\"'`;:-()&$", lower=True, split=" "
    )
    for word in words:
        for i in range(0, len(word) - 1):
            bigram = str(word[i : i + 2]).lower()
            if bigram in bigrams_dict:
                bigram_counts[bigram] = bigram_counts[bigram] + 1
                total_count = total_count + 1

    bigrams_frequency = []
    for bigram in bigrams:
        bigrams_frequency.append(float(bigram_counts[bigram] / total_count))

    return bigrams_frequency


def most_common_letter_trigrams(input_text: str) -> List[float]:
    """
    Calculates the frequency of occurrence of the most common letter trigrams in the input text.

    :param input_text: A string representing the input text.
    :return A list of floats representing the frequency of 
        occurrence of the most common letter trigrams in the input text.
    """
    trigrams = [
        "the",
        "and",
        "ing",
        "her",
        "hat",
        "his",
        "tha",
        "ere",
        "for",
        "ent",
        "ion",
        "ter",
        "was",
        "you",
        "ith",
        "ver",
        "all",
        "wit",
        "thi",
        "tio",
    ]
    trigrams_dict = {}
    for t in trigrams:
        trigrams_dict[t] = True

    trigram_counts = {}
    for t in trigrams_dict:
        trigram_counts[t] = 0

    total_count = 0
    words = text.text_to_word_sequence(
        input_text, filters=",.?!\"'`;:-()&$", lower=True, split=" "
    )
    for word in words:
        for i in range(0, len(word) - 2):
            trigram = str(word[i : i + 3]).lower()
            if trigram in trigrams_dict:
                trigram_counts[trigram] = trigram_counts[trigram] + 1
                total_count = total_count + 1

    trigrams_frequency = []
    for t in trigrams:
        trigrams_frequency.append(float(trigram_counts[t] / total_count))

    return trigrams_frequency


def digits_percentage(input_text: str) -> float:
    """
    Calculates the percentage of digits out of total characters in the input text.

    :param input_text: A string representing the input text.
    :return A float representing the percentage of digits out of total characters in the input text.
    """
    input_text = input_text.lower()
    chars_count = len(str(input_text))
    digits_count = list([1 for i in str(input_text) if i.isnumeric() == True]).count(1)
    return digits_count / chars_count


def characters_percentage(input_text: str) -> float:
    """
    Calculates the percentage of characters out of total characters in the input text.

    :param input_text: A string representing the input text.
    :return A float representing the percentage of characters out of total characters in the input text.
    """
    input_text = input_text.lower().replace(" ", "")
    characters = "abcdefghijklmnopqrstuvwxyz"
    all_chars_count = len(str(input_text))
    chars_count = list([1 for i in str(input_text) if i in characters]).count(1)
    return chars_count / all_chars_count


def upper_case_characters_percentage(input_text: str) -> float:
    """
    Calculates the percentage of uppercase characters out of total characters in the input text.

    :param input_text: A string representing the input text.
    :return A float representing the percentage of uppercase characters out of total characters in the input text.
    """
    input_text = input_text.replace(" ", "")
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    all_chars_count = len(str(input_text))
    chars_count = list([1 for i in str(input_text) if i in characters]).count(1)
    return chars_count / all_chars_count


def frequency_of_digits(input_text: str) -> List[float]:
    """
    Calculates the frequency of digits in the input text.

    :param input_text: A string representing the input text.
    :return A list of floats representing the frequency of digits in the input text.
    """
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    digits_counts = {}
    for digit in digits:
        digits_counts[str(digit)] = 0

    all_digits = re.findall("\d", input_text)
    for digit in all_digits:
        digits_counts[digit] += 1

    digits_counts = SortedDict(digits_counts)
    digits_counts = np.array(digits_counts.values())
    return np.divide(digits_counts, characters_count(input_text))


def frequency_of_digits_numbers(input_text: str, digit_length: int) -> float:
    """
    Calculates the frequency of digits of a specific length in the input text.

    :param input_text: A string representing the input text.
    :param digit_length: An integer representing the length of the digits to be counted.
    :return A float representing the frequency of digits of a specific length in the input text.
    """
    input_text = str(input_text).lower()
    words = text.text_to_word_sequence(
        input_text,
        filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"',
        lower=True,
        split=" ",
    )

    count = 0
    word_count = len(words)
    for w in words:
        if w.isnumeric() == True and len(w) == digit_length:
            count = count + 1

    return count / word_count


def frequency_of_word_length(input_text: str) -> List[float]:
    """
    Calculates the frequency of words of specific lengths up to 15 in the input text.

    :param input_text: A string representing the input text.
    :return A list of floats representing the frequency of words of specific lengths up to 15 in the input text.
    """
    lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    word_length_frequencies = {}
    for l in lengths:
        word_length_frequencies[l] = 0

    words = text.text_to_word_sequence(
        input_text,
        filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"',
        lower=True,
        split=" ",
    )
    for w in words:
        word_length = len(w)
        if word_length in word_length_frequencies:
            word_length_frequencies[word_length] = (
                word_length_frequencies[word_length] + 1
            )

    frequency_vector = [0] * (len(lengths))
    total_count = sum(list(word_length_frequencies.values()))
    for w in word_length_frequencies:
        frequency_vector[w - 1] = word_length_frequencies[w] / total_count

    return frequency_vector


def frequency_of_special_characters(input_text: str) -> List[float]:
    """
    Calculates the frequency of special characters in the input text.

    :param input_text: A string representing the input text.
    :return A list of floats representing the frequency of special characters in the input text.
    """
    input_text = str(input_text).lower()
    special_characters = open(
        "writeprint_resources/writeprints_special_chars.txt", "r"
    ).readlines()
    special_characters = [s.strip("\n") for s in special_characters]
    special_characters_frequency_dict = {}
    for c in range(0, len(special_characters)):
        special_char = special_characters[c]
        special_characters_frequency_dict[special_char] = 0
        for i in str(input_text):
            if special_char == i:
                special_characters_frequency_dict[special_char] = (
                    special_characters_frequency_dict[special_char] + 1
                )

    # making a vector out of the frequencies
    vector_of_frequencies = [0] * len(special_characters)
    total_count = sum(list(special_characters_frequency_dict.values())) + 1
    for c in range(0, len(special_characters)):
        special_char = special_characters[c]
        vector_of_frequencies[c] = (
            special_characters_frequency_dict[special_char] / total_count
        )

    vector_of_frequencies = np.array(vector_of_frequencies)
    return vector_of_frequencies


def function_words_percentage(input_text: str) -> List[int]:
    """
    Calculates the frequency of function words in the input text.

    :param input_text: A string representing the input text.
    :return A list of integers representing the frequency of function words in the input text.
    """
    function_words = open(
        cur_dir_path + "/writeprint_resources/function_word.txt", "r"
    ).readlines()
    function_words = [f.strip("\n") for f in function_words]
    words = text.text_to_word_sequence(
        input_text,
        filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"',
        lower=True,
        split=" ",
    )
    frequency_of_function_words = []
    for i in range(len(function_words)):
        function_word = function_words[i]
        freq = 0
        for word in words:
            if word == function_word:
                freq += 1
        frequency_of_function_words.append(freq)

    return frequency_of_function_words


def frequency_of_punctuation_characters(input_text: str) -> List[float]:
    """
    Calculates the frequency of punctuation characters in the input text.

    :param input_text: A string representing the input text.
    :return A list of floats representing the frequency of punctuation characters in the input text.
    """
    input_text = str(input_text).lower().replace(" ", "")
    special_characters = open(
        cur_dir_path + "/writeprint_resources/writeprints_punctuation.txt", "r"
    ).readlines()
    special_characters = [s.strip("\n") for s in special_characters]
    special_characters_frequency_dict = {}
    for c in range(0, len(special_characters)):
        special_char = special_characters[c]
        special_characters_frequency_dict[special_char] = 0
        for i in str(input_text):
            if special_char == i:
                special_characters_frequency_dict[special_char] = (
                    special_characters_frequency_dict[special_char] + 1
                )

    # making a vector out of the frequencies
    vector_of_frequencies = [0] * len(special_characters)
    total_count = sum(list(special_characters_frequency_dict.values())) + 1
    for c in range(0, len(special_characters)):
        special_char = special_characters[c]
        vector_of_frequencies[c] = (
            special_characters_frequency_dict[special_char] / total_count
        )

    return vector_of_frequencies


def misspellings_percentage(input_text: str) -> float:
    """
    Calculates the percentage of misspelled words in the input text.

    :param input_text: A string representing the input text.
    :return A float representing the percentage of misspelled words in the input text.
    """
    misspelled_words = open(
        cur_dir_path + "/writeprint_resources/writeprints_misspellings.txt", "r"
    ).readlines()
    misspelled_words = [f.strip("\n") for f in misspelled_words]
    words = text.text_to_word_sequence(
        input_text,
        filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"',
        lower=True,
        split=" ",
    )
    misspelled_words_intersection = set(words).intersection(set(misspelled_words))
    return len(misspelled_words_intersection) / len(list(words))


def legomena(input_text: str) -> List[float]:
    """
    Calculates the ratio of hapax legomena and dis legomena in the input text.

    :param input_text: A string representing the input text.
    :return A list of floats representing the ratio of hapax legomena and dis legomena in the input text.
    """
    freq = nltk.FreqDist(word for word in input_text.split())
    hapax = [key for key, val in freq.items() if val == 1]
    dis = [key for key, val in freq.items() if val == 2]
    try:
        return list(
            (len(hapax) / len(input_text.split()), len(dis) / len(input_text.split()))
        )
    except:
        return [0, 0]


def pos_tag_frequency(input_text: str) -> List[float]:
    """
    Calculates the frequency of each part-of-speech tag in the input text.

    :param input_text: A string representing the input text.
    :return A list of floats representing the frequency of 
        each part-of-speech tag in the input text.
    """
    pos_tags = []
    doc = nlp(str(input_text))
    for token in doc:
        pos_tags.append(str(token.pos_))

    tagset = [
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CONJ",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "SPACE",
        "X",
    ]
    tags = [tag for tag in pos_tags]
    return [tags.count(tag) / len(tags) for tag in tagset]


def total_words(input_text: str) -> int:
    """
    Calculates the total number of words in the input text.

    :param input_text: A string representing the input text.
    :return An integer representing the total number of words in the input text.
    """
    words = text.text_to_word_sequence(
        input_text, filters=",.?!\"'`;:-()&$", lower=True, split=" "
    )
    return len(words)


def average_word_length(input_text: str) -> float:
    """
    Calculates the average length of words in the input text.

    :param input_text: A string representing the input text.
    :return A float representing the average length of words in the input text.
    """
    words = text.text_to_word_sequence(
        input_text, filters=",.?!\"'`;:-()&$", lower=True, split=" "
    )
    lengths = []
    for word in words:
        lengths.append(len(word))
    return np.mean(lengths)


def no_of_short_words(input_text: str) -> int:
    """
    Calculates the number of short words (3 characters or less) in the input text.

    :param input_text: A string representing the input text.
    :return An integer representing the number of short words in the input text.
    """
    words = text.text_to_word_sequence(
        input_text, filters=",.?!\"'`;:-()&$", lower=True, split=" "
    )
    short_words = []
    for word in words:
        if len(word) <= 3:
            short_words.append(word)
    return len(short_words)


def calculate_features(input_text: str) -> List[float]:
    """
    Calculates a list of features for the input text.

    :param input_text: A string representing the input text.
    :return A list of floats representing the features of the input text.
    """
    feature_list = []

    feature_list.extend([total_words(input_text)])
    feature_list.extend([average_word_length(input_text)])
    feature_list.extend([no_of_short_words(input_text)])
    feature_list.extend([characters_percentage(input_text)])
    feature_list.extend([digits_percentage(input_text)])
    feature_list.extend([upper_case_characters_percentage(input_text)])
    feature_list.extend(frequency_of_special_characters(input_text))
    feature_list.extend(frequency_of_letters(input_text))
    feature_list.extend(frequency_of_digits(input_text))
    feature_list.extend(most_common_letter_bigrams(input_text))
    feature_list.extend(most_common_letter_trigrams(input_text))
    feature_list.extend(legomena(input_text))
    feature_list.extend(function_words_percentage(input_text))
    feature_list.extend(pos_tag_frequency(input_text))
    feature_list.extend(frequency_of_punctuation_characters(input_text))

    return feature_list
