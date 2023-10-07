def warn(*args, **kwargs):
    pass


import itertools
import warnings
from collections import Counter

from sortedcontainers import SortedDict

warnings.warn = warn
import re

import nltk
import numpy as np
import spacy
from keras.preprocessing import text

nlp = spacy.load("en_core_web_sm")

from typing import Dict, List


def total_words(input_text: str) -> int:
    """
    Returns the total number of words in the input text 
    after removing punctuations and unifying case.

    Args:
    input_text (str): The input text to count the words from.

    Returns:
    int: The total number of words in the input text.
    """
    words = text.text_to_word_sequence(
        input_text, filters=",.?!\"'`;:-()&$", lower=True, split=" "
    )
    return len(words)


def unique_words_count(input_text: str) -> float:
    """
    Calculates the ratio of unique words to 
    total words in the input text.

    Args:
    input_text (str): The input text to calculate 
        the ratio of unique words to total words.

    Returns:
    float: The ratio of unique words to total
        words in the input text.
    """
    words = text.text_to_word_sequence(
        input_text, filters=",.?!\"'`;:-()&$", lower=True, split=" "
    )
    return len(list(set(words))) / total_words(input_text)


def larger_words_count(input_text: str) -> float:
    """
    Calculates the ratio of words in the input 
    text that have a length greater than 4.

    Args:
    input_text (str): The input text to be analyzed.

    Returns:
    float: The ratio of words in the input 
        text that have a length greater than 4.
    """
    words = text.text_to_word_sequence(
        input_text,
        filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"',
        lower=True,
        split=" ",
    )
    short_words = []
    for word in words:
        if len(word) > 4:
            short_words.append(word)
    return len(short_words) / total_words(input_text)


def average_characters_per_word(input_text: str) -> float:
    """
    Calculates the average number of characters 
    per word in the input text.

    Args:
    input_text (str): The input text to calculate 
        the average characters per word for.

    Returns:
    float: The average number of characters per
        word in the input text.
    """
    return character_count(input_text) / total_words(input_text)


def character_count(input_text: str) -> int:
    """
    Counts the number of characters in the input text.

    Args:
    input_text (str): The input text to count characters from.

    Returns:
    int: The number of characters in the input text.
    """
    input_text = input_text.lower()
    return len(input_text)


def letters(input_text: str) -> List[float]:
    """
    This function takes a string as input and returns a list 
    of floats representing the frequency of each letter in the input string.

    Args:
    input_text (str): The input string for which the 
        frequency of each letter needs to be calculated.

    Returns:
    List[float]: A list of floats representing the 
        frequency of each letter in the input string.
    """
    input_text = str(input_text).lower()  # because its case sensitive
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
    totalCount = character_count(input_text)
    for c in range(0, len(characters)):
        char = characters[c]
        vector_of_frequencies[c] = chars_frequency_dict[char] / totalCount

    return vector_of_frequencies


def most_common_letter_bigrams(input_text: str) -> np.ndarray:
    """
    This function takes in a string of text and returns 
    the 50 most common letter bigrams in the text.

    Args:
    input_text (str): The input text to analyze

    Returns:
    np.ndarray: An array of the 50 most common letter 
        bigrams in the text, sorted in descending order of frequency.
    """

    characters = "abcdefghijklmnopqrstuvwxyz"

    bigrams = []
    tagsets = list(itertools.combinations(characters, 2))
    for tagset in tagsets:
        bigrams.append("".join(tagset))
    bigram_counts = {}
    for t in bigrams:
        bigram_counts[t] = 0

    total_count = 0
    words = text.text_to_word_sequence(
        input_text, filters=",.?!\"'`;:-()&$", lower=True, split=" "
    )
    for word in words:
        for i in range(0, len(word) - 1):
            bigram = str(word[i : i + 2]).lower()
            if bigram in bigrams:
                bigram_counts[bigram] = bigram_counts[bigram] + 1
                total_count = total_count + 1

    bigram_counts = SortedDict(bigram_counts)
    bigram_counts = np.array(bigram_counts.values())
    bigram_counts = sorted(bigram_counts, reverse=True)
    bigram_counts = bigram_counts[:50]
    return np.divide(bigram_counts, character_count(input_text))


def most_common_letter_trigrams(input_text: str) -> List[float]:
    """
    This function takes a string as input_text and returns 
    a list of 50 most common letter trigrams in the input_text.
    The function first generates all possible letter trigrams 
    and then counts the frequency of each trigram in the input_text.
    The function then returns the 50 most common trigrams as a 
    list of floats, where each float represents the frequency of the trigram
    divided by the total number of characters in the input_text.

    Args:
    input_text (str): The input text for which the most 
        common letter trigrams are to be found.

    Returns:
    List[float]: A list of 50 most common letter trigrams 
        in the input_text, where each float represents the frequency of the trigram
        divided by the total number of characters in the input_text.
    """

    characters = "abcdefghijklmnopqrstuvwxyz"

    trigrams = []
    tagsets = list(itertools.combinations(characters, 3))
    for tagset in tagsets:
        trigrams.append("".join(tagset))

    trigram_counts = {}
    for t in trigrams:
        trigram_counts[t] = 0

    total_count = 0
    words = text.text_to_word_sequence(
        input_text, filters=",.?!\"'`;:-()&$", lower=True, split=" "
    )
    for word in words:
        for i in range(0, len(word) - 2):
            trigram = str(word[i : i + 3]).lower()
            if trigram in trigrams:
                trigram_counts[trigram] = trigram_counts[trigram] + 1
                total_count = total_count + 1

    trigram_counts = SortedDict(trigram_counts)
    trigram_counts = np.array(trigram_counts.values())
    trigram_counts = sorted(trigram_counts, reverse=True)
    trigram_counts = trigram_counts[:50]
    return np.divide(trigram_counts, character_count(input_text))


def digits(input_text: str) -> List[float]:
    """
    Calculates the frequency of digits in the input text and 
    returns a list of floats representing the normalized counts.

    Args:
    - input_text (str): The input text to analyze

    Returns:
    - List[float]: A list of floats representing the 
        normalized counts of digits in the input text
    """
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    digits_counts: Dict[str, int] = {}
    for digit in digits:
        digits_counts[str(digit)] = 0

    all_digits = re.findall("\d", input_text)
    for digit in all_digits:
        digits_counts[digit] += 1

    digits_counts = SortedDict(digits_counts)
    digits_counts = np.array(digits_counts.values())
    return np.divide(digits_counts, character_count(input_text))


def two_digit_numbers(input_text: str) -> np.ndarray:
    """
    Returns an array of the counts of two-digit numbers 
    in the input text, divided by the character count of the input text.

    Args:
    input_text (str): The input text to count two-digit numbers from.

    Returns:
    np.ndarray: An array of the counts of two-digit numbers
        in the input text, divided by the character count of the input text.
    """
    digits = []
    for i in range(10, 100):
        digits.append(i)
    digits_counts = {}
    for digit in digits:
        digits_counts[str(digit)] = 0

    all_digits = re.findall("\d\d", input_text)
    for digit in all_digits:
        try:
            digits_counts[digit] += 1
        except:
            pass
    digits_counts = SortedDict(digits_counts)
    digits_counts = np.array(digits_counts.values())
    digits_counts = sorted(digits_counts, reverse=True)
    digits_counts = digits_counts[:50]
    return np.divide(digits_counts, character_count(input_text))


def three_digit_numbers(input_text: str) -> np.ndarray:
    """
    This function takes a string as input_text and returns an 
    array of the top 50 most frequently occurring three-digit numbers
    in the input_text, divided by the total number of characters in the input_text.

    Args:
    input_text (str): The input text to search for three-digit numbers.

    Returns:
    np.ndarray: An array of the top 50 most frequently occurring
        three-digit numbers in the input_text, divided by 
        the total number of characters in the input_text.
    """
    digits = []
    for i in range(100, 1000):
        digits.append(i)
    digits_counts = {}
    for digit in digits:
        digits_counts[str(digit)] = 0

    all_digits = re.findall("\d\d\d", input_text)
    for digit in all_digits:
        try:
            digits_counts[digit] += 1
        except:
            pass
    digits_counts = SortedDict(digits_counts)
    digits_counts = np.array(digits_counts.values())
    digits_counts = sorted(digits_counts, reverse=True)
    digits_counts = digits_counts[:50]
    return np.divide(digits_counts, character_count(input_text))


def word_lengths(input_text: str) -> List[int]:
    """
    This function takes a string as input and returns a list of 
    integers representing the frequency of words of different lengths in the input text.

    Args:
    input_text (str): The input text for which the word length 
        frequencies are to be calculated.

    Returns:
    List[int]: A list of integers representing the frequency of 
        words of different lengths in the input text.
    """

    lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    word_length_frequencies = {}
    for l in lengths:
        word_length_frequencies[l] = 0

    words = text.text_to_word_sequence(
        input_text, filters=",.?!\"'`;:-()&$", lower=False, split=" "
    )
    for w in words:
        wordLength = len(w)
        if wordLength in word_length_frequencies:
            word_length_frequencies[wordLength] = (
                word_length_frequencies[wordLength] + 1
            )

    word_length_frequencies = SortedDict(word_length_frequencies)
    return word_length_frequencies.values()


def frequency_of_special_characters(input_text: str) -> np.array:
    """
    Calculates the frequency of special characters in the 
    input text and returns a numpy array of the frequencies.

    Args:
    input_text (str): The input text for which the 
        frequency of special characters is to be calculated.

    Returns:
    np.array: A numpy array of the frequencies of 
        special characters in the input text.
    """
    input_text = input_text.lower()
    with open("writeprint_resources/writeprints_special_chars.txt", "r") as f:
        special_characters = [line.strip() for line in f.readlines()]

    special_characters_frequency_dict = {char: 0 for char in special_characters}

    for char in input_text:
        if char in special_characters_frequency_dict:
            special_characters_frequency_dict[char] += 1

    total_count = sum(special_characters_frequency_dict.values()) + 1
    vector_of_frequencies = [
        special_characters_frequency_dict[char] / total_count
        for char in special_characters
    ]

    return np.array(vector_of_frequencies)


def function_words_percentage(input_text: str) -> np.array:
    """
    Calculates the percentage of function words in the input text.

    Args:
    input_text (str): The input text to calculate the percentage of function words.

    Returns:
    np.array: An array of the percentage of function words in the input text.
    """

    with open("writeprint_resources/function_word.txt", "r") as f:
        function_words = [line.strip() for line in f.readlines()]

    words = text.text_to_word_sequence(
        input_text, filters=",.?!\"'`;:-()&$", lower=True, split=" "
    )
    frequency_of_function_words = []

    for function_word in function_words:
        freq = words.count(function_word)
        frequency_of_function_words.append(freq)

    frequency_of_function_words = np.array(frequency_of_function_words)
    frequency_of_function_words = sorted(frequency_of_function_words, reverse=True)
    frequency_of_function_words = frequency_of_function_words[:50]

    return np.divide(frequency_of_function_words, total_words(input_text))


def frequency_of_punctuation_characters(input_text: str) -> list:
    """
    Calculates the frequency of punctuation characters 
    in the input text and returns a vector of frequencies.

    Args:
    input_text (str): The input text to calculate the 
        frequency of punctuation characters.

    Returns:
    list: A list of frequencies of punctuation characters in the input text.
    """

    input_text = str(input_text).lower()
    input_text = input_text.replace(" ", "")
    with open("writeprint_resources/writeprints_punctuation.txt", "r") as file:
        special_characters = [s.strip() for s in file.readlines()]

    special_characters_frequency_dict = {}

    for special_char in special_characters:
        special_characters_frequency_dict[special_char] = 0
        for char in input_text:
            if special_char == char:
                special_characters_frequency_dict[special_char] += 1

    # Making a vector out of the frequencies
    vector_of_frequencies = [0] * len(special_characters)
    total_count = sum(special_characters_frequency_dict.values()) + 1
    for c, special_char in enumerate(special_characters):
        vector_of_frequencies[c] = (
            special_characters_frequency_dict[special_char] / total_count
        )

    return vector_of_frequencies


def pos_tag_frequency(input_text):
    """
    Calculate the frequency of POS tags for the given text.

    Args:
    - input_text (str): The input text to calculate POS tag frequencies for.

    Returns:
    - list: A list of frequencies corresponding to the tagset.
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
    return list(tuple(tags.count(tag) / len(tags) for tag in tagset))


def pos_tag_bigram_frequency(input_text):
    """
    Calculate the frequency of POS tag bigrams for the given text.

    Args:
    - input_text (str): The input text to calculate POS 
        tag bigram frequencies for.

    Returns:
    - list[float]: A list of the 50 highest bigram 
        frequencies corresponding to the POS tagset.
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
    tagset = list(itertools.combinations(tagset, 2))
    tagset_freq = [0] * len(tagset)

    for i in range(1, len(pos_tags)):
        if (pos_tags[i - 1], pos_tags[i]) in tagset:
            tagset_freq[tagset.index((pos_tags[i - 1], pos_tags[i]))] += 1

    final = [(val / (len(pos_tags))) for val in tagset_freq]
    final = sorted(final, reverse=True)
    return final[:50]


def pos_tag_trigram_frequency(input_text):
    """
    Calculate the frequency of POS tag trigrams for the given text.

    Args:
    - input_text (str): The input text to calculate POS 
        tag trigram frequencies for.

    Returns:
    - list[float]: A list of the 50 highest trigram 
        frequencies corresponding to the POS tagset.
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
    tagset = list(itertools.combinations(tagset, 3))
    tagset_freq = [0] * len(tagset)

    for i in range(2, len(pos_tags)):
        if (pos_tags[i - 2], pos_tags[i - 1], pos_tags[i]) in tagset:
            tagset_freq[
                tagset.index((pos_tags[i - 2], pos_tags[i - 1], pos_tags[i]))
            ] += 1

    final = [(val / (len(pos_tags))) for val in tagset_freq]
    final = sorted(final, reverse=True)
    return final[:50]


def unigram_words(input_text: str) -> List[int]:
    """
    Returns the frequency of the top 50 most common words in the input text.

    Args:
    input_text (str): The input text to analyze.

    Returns:
    List[int]: A list of the frequency of the top 
        50 most common words in the input text.
    """
    words = text.text_to_word_sequence(
        input_text, filters=",.?!\"'`;:-()&$", lower=True, split=" "
    )
    values = Counter(words).values()
    values = sorted(values, reverse=True)

    return values[:50]


def bigram_words(input_text):
    """
    Calculate the frequency of word bigrams for the given text.

    Args:
    - input_text (str): The input text for which to 
        calculate word bigram frequencies.

    Returns:
    - list[float]: A list of the 50 highest bigram 
        frequencies corresponding to the words in the input text.
    """

    words = text.text_to_word_sequence(
        input_text, filters=",.?!\"'`;:-()&$", lower=True, split=" "
    )
    bigrams = list(set(list(nltk.bigrams(words))))

    freq_list = [0] * len(bigrams)
    for i in range(1, len(words)):
        if (words[i - 1], words[i]) in bigrams:
            freq_list[bigrams.index((words[i - 1], words[i]))] += 1

    values = [(val / (len(words))) for val in freq_list]
    values = sorted(values, reverse=True)

    return values[:50]


def trigram_words(input_text):
    """
    Calculate the frequency of word trigrams for the given text.

    Args:
    - input_text (str): The input text for which to
        calculate word trigram frequencies.

    Returns:
    - list[float]: A list of the 50 highest trigram 
        frequencies corresponding to the words in the input text.
    """

    words = text.text_to_word_sequence(
        input_text, filters=",.?!\"'`;:-()&$", lower=True, split=" "
    )
    trigrams = list(set(list(nltk.trigrams(words))))

    freq_list = [0] * len(trigrams)
    for i in range(2, len(words)):
        if (words[i - 2], words[i - 1], words[i]) in trigrams:
            freq_list[trigrams.index((words[i - 2], words[i - 1], words[i]))] += 1

    values = [(val / (len(words))) for val in freq_list]
    values = sorted(values, reverse=True)

    return values[:50]


def misspellings_percentage(input_text):
    """
    Calculate the percentage of misspelled words from 
    a predefined list present in the given text.

    Args:
    - input_text (str): The input text for which to 
        calculate the misspellings percentage.

    Returns:
    - list[float]: A list of the percentages of the 
        50 highest frequent misspelled words corresponding to the words in the input text.
    """

    misspelledWords = open(
        "writeprint_resources/writeprints_misspellings.txt", "r"
    ).readlines()
    misspelledWords = [f.strip("\n") for f in misspelledWords]

    words = text.text_to_word_sequence(
        input_text, filters=",.?!\"'`;:-()&$", lower=True, split=" "
    )
    frequency_of_function_words = []
    for i in range(len(misspelledWords)):
        functionWord = misspelledWords[i]
        freq = 0
        for word in words:
            if word == functionWord:
                freq += 1
        frequency_of_function_words.append(freq)

    frequency_of_function_words = np.array(frequency_of_function_words)
    frequency_of_function_words = sorted(frequency_of_function_words, reverse=True)
    frequency_of_function_words = frequency_of_function_words[:50]
    return np.divide(frequency_of_function_words, total_words(input_text))


def digits_percentage(input_text: str) -> float:
    """
    Calculates the percentage of digits in the input text.

    Args:
    input_text (str): The input text to 
        calculate the percentage of digits.

    Returns:
    float: The percentage of digits in the input text.
    """
    charsCount = character_count(input_text)
    digitsCount = len(re.findall("\d", input_text))
    return digitsCount / charsCount


def letters_percentage(input_text: str) -> float:
    """
    Calculates the percentage of letters in the given input text.

    Args:
    input_text (str): The input text to calculate the percentage of letters from.

    Returns:
    float: The percentage of letters in the input text.
    """
    lettersCount = len(re.findall("[A-Za-z]", input_text))
    charsCount = character_count(input_text)
    return lettersCount / charsCount


def uppercase_letters_percentage(input_text: str) -> float:
    """
    Calculates the percentage of uppercase letters in the given input text.

    Args:
    input_text (str): The input text to calculate 
        the percentage of uppercase letters from.

    Returns:
    float: The percentage of uppercase letters in the input text.
    """
    lettersCount = len(re.findall("[A-Z]", input_text))
    charsCount = len(input_text)
    return lettersCount / charsCount


def hapax_legomenon(input_text: str) -> list:
    """
    Calculates the hapax legomenon ratio of the input text.

    Args:
    input_text (str): The input text to calculate hapax legomenon ratio for.

    Returns:
    list: A list containing the hapax legomenon ratio of the input text.

    """
    freq = nltk.FreqDist(word for word in input_text.split())
    hapax = [key for key, val in freq.items() if val == 1]

    try:
        return list(len(hapax) / total_words(input_text))
    except:
        return [0]


def calculate_features(input_text: str) -> List[float]:
    """
    Calculate a variety of linguistic features for the given text.

    Args:
        input_text (str): The input text for which to calculate the linguistic features.

    Returns:
        list[float]: A list of linguistic feature values corresponding to the input text.
    """

    feature_list = []

    feature_list.extend([total_words(input_text)])
    feature_list.extend([character_count(input_text)])
    feature_list.extend([average_characters_per_word(input_text)])
    feature_list.extend(letters(input_text))
    feature_list.extend(most_common_letter_bigrams(input_text))
    feature_list.extend(most_common_letter_trigrams(input_text))
    feature_list.extend(digits(input_text))
    feature_list.extend(two_digit_numbers(input_text))
    feature_list.extend(three_digit_numbers(input_text))
    feature_list.extend(word_lengths(input_text))
    feature_list.extend(frequency_of_special_characters(input_text))
    feature_list.extend(function_words_percentage(input_text))
    feature_list.extend(frequency_of_punctuation_characters(input_text))
    feature_list.extend(pos_tag_frequency(input_text))
    feature_list.extend(pos_tag_bigram_frequency(input_text))
    feature_list.extend(pos_tag_trigram_frequency(input_text))
    feature_list.extend(unigram_words(input_text))
    feature_list.extend(bigram_words(input_text))
    feature_list.extend(trigram_words(input_text))
    feature_list.extend(misspellings_percentage(input_text))
    feature_list.append(unique_words_count(input_text))
    feature_list.extend([digits_percentage(input_text)])
    feature_list.extend([letters_percentage(input_text)])
    feature_list.extend([uppercase_letters_percentage(input_text)])
    feature_list.extend(hapax_legomenon(input_text))

    return feature_list
