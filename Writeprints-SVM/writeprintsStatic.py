def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
import nltk
import spacy
import re
from sortedcontainers import SortedDict
import os
from keras.preprocessing import text

nlp = spacy.load('en_core_web_sm')
cur_dir_path = os.getcwd() + "/"  # + "writeprints_experiments/"

############## FEATURES COMPUTATION #####################
def getCleanText(inputText):
    # cleanText = text.text_to_word_sequence(inputText,filters='!#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"\' ', lower=False, split=" ")
    cleanText = text.text_to_word_sequence(inputText, filters='', lower=False, split=" ")

    cleanText = ''.join(str(e) + " " for e in cleanText)
    return cleanText


def charactersCount(inputText):
    '''
    Calculates character count including spaces
    '''
    inputText = inputText.lower()
    charCount = len(str(inputText))
    return charCount


def averageCharacterPerWord(inputText):
    '''
    Calculates average number of characters per word
    '''

    words = text.text_to_word_sequence(inputText, filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"', lower=True, split=" ")
    inputText = inputText.lower().replace(" ", "")
    charCount = len(str(inputText))

    avgCharCount = charCount / len(words)
    return avgCharCount


def frequencyOfLetters(inputText):
    '''
    Calculates the frequency of letters
    '''

    inputText = str(inputText).lower()  # because its case sensitive
    inputText = inputText.lower().replace(" ", "")
    characters = "abcdefghijklmnopqrstuvwxyz"
    charsFrequencyDict = {}
    for c in range(0, len(characters)):
        char = characters[c]
        charsFrequencyDict[char] = 0
        for i in str(inputText):
            if char == i:
                charsFrequencyDict[char] = charsFrequencyDict[char] + 1

    # making a vector out of the frequencies
    vectorOfFrequencies = [0] * len(characters)
    totalCount = sum(list(charsFrequencyDict.values()))
    for c in range(0, len(characters)):
        char = characters[c]
        vectorOfFrequencies[c] = charsFrequencyDict[char] / totalCount

    return vectorOfFrequencies


def mostCommonLetterBigrams(inputText):
    # to do


    bigrams = ['th','he','in','er','an','re','nd','at','on','nt','ha','es','st' ,'en','ed','to','it','ou','ea','hi','is','or','ti','as','te','et' ,'ng','of','al','de','se','le','sa','si','ar','ve','ra','ld','ur']

    bigramsD = {}
    for t in bigrams:
        bigramsD[t] = True

    bigramCounts = {}
    for t in bigramsD:
        bigramCounts[t] = 0

    totalCount = 0
    words = text.text_to_word_sequence(inputText, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
    for word in words:
        for i in range(0, len(word) - 1):
            bigram = str(word[i:i + 2]).lower()
            if bigram in bigrams:
                bigramCounts[bigram] = bigramCounts[bigram] + 1
                totalCount = totalCount + 1

    bigramsFrequency = []
    for t in bigrams:
        bigramsFrequency.append(float(bigramCounts[t] / totalCount))

    return bigramsFrequency


def mostCommonLetterTrigrams(inputText):
    # to do
    trigrams = ["the", "and", "ing", "her", "hat", "his", "tha", "ere", "for", "ent", "ion", "ter", "was", "you", "ith",
                "ver", "all", "wit", "thi", "tio"]
    trigramsD = {}
    for t in trigrams:
        trigramsD[t] = True

    trigramCounts = {}
    for t in trigramsD:
        trigramCounts[t] = 0

    totalCount = 0
    words = text.text_to_word_sequence(inputText, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
    for word in words:
        for i in range(0, len(word) - 2):
            trigram = str(word[i:i + 3]).lower()
            if trigram in trigrams:
                trigramCounts[trigram] = trigramCounts[trigram] + 1
                totalCount = totalCount + 1

    trigramsFrequency = []
    for t in trigrams:
        trigramsFrequency.append(float(trigramCounts[t] / totalCount))

    return trigramsFrequency


def digitsPercentage(inputText):
    '''
    Calculates the percentage of digits out of total characters
    '''
    inputText = inputText.lower()
    charsCount = len(str(inputText))
    digitsCount = list([1 for i in str(inputText) if i.isnumeric() == True]).count(1)
    return digitsCount / charsCount


def charactersPercentage(inputText):
    '''
    Calculates the percentage of characters out of total characters
    '''

    inputText = inputText.lower().replace(" ", "")
    characters = "abcdefghijklmnopqrstuvwxyz"
    allCharsCount = len(str(inputText))
    charsCount = list([1 for i in str(inputText) if i in characters]).count(1)
    return charsCount / allCharsCount


def upperCaseCharactersPercentage(inputText):
    '''
    Calculates the percentage of uppercase characters out of total characters
    '''

    inputText = inputText.replace(" ", "")
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    allCharsCount = len(str(inputText))
    charsCount = list([1 for i in str(inputText) if i in characters]).count(1)
    return charsCount / allCharsCount


def frequencyOfDigits(inputText):
    '''
    Calculates the frequency of digits
    '''

    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    digitsCounts = {}
    for digit in digits:
        digitsCounts[str(digit)] = 0

    alldigits = re.findall('\d', inputText)
    for digit in alldigits:
        digitsCounts[digit] += 1

    digitsCounts = SortedDict(digitsCounts)
    digitsCounts = np.array(digitsCounts.values())
    return np.divide(digitsCounts, charactersCount(inputText))

def frequencyOfDigitsNumbers(inputText, digitLength):
    '''
    Calculates the frequency of digits
    '''

    inputText = str(inputText).lower()  # because its case sensitive
    words = text.text_to_word_sequence(inputText, filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"', lower=True, split=" ")

    count = 0
    wordCount = len(words)
    for w in words:
        if w.isnumeric() == True and len(w) == digitLength:
            count = count + 1

    return count / wordCount


def frequencyOfWordLength(inputText):
    '''
    Calculate frequency of words of specific lengths upto 15
    '''
    lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    wordLengthFrequencies = {}
    for l in lengths:
        wordLengthFrequencies[l] = 0

    words = text.text_to_word_sequence(inputText, filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"', lower=True, split=" ")
    for w in words:
        wordLength = len(w)
        if wordLength in wordLengthFrequencies:
            wordLengthFrequencies[wordLength] = wordLengthFrequencies[wordLength] + 1

    frequencyVector = [0] * (len(lengths))
    totalCount = sum(list(wordLengthFrequencies.values()))
    for w in wordLengthFrequencies:
        frequencyVector[w - 1] = wordLengthFrequencies[w] / totalCount

    return frequencyVector


def frequencyOfSpecialCharacters(inputText):


    inputText = str(inputText).lower()  # because its case insensitive
    # inputText = inputText.lower().replace(" ", "")
    specialCharacters = open("writeprintresources/writeprints_special_chars.txt", "r").readlines()
    specialCharacters = [s.strip("\n") for s in specialCharacters]
    specialCharactersFrequencyDict = {}
    for c in range(0, len(specialCharacters)):
        specialChar = specialCharacters[c]
        specialCharactersFrequencyDict[specialChar] = 0
        for i in str(inputText):
            if specialChar == i:
                specialCharactersFrequencyDict[specialChar] = specialCharactersFrequencyDict[specialChar] + 1


    # making a vector out of the frequencies
    vectorOfFrequencies = [0] * len(specialCharacters)
    totalCount = sum(list(specialCharactersFrequencyDict.values())) + 1
    for c in range(0, len(specialCharacters)):
        specialChar = specialCharacters[c]
        vectorOfFrequencies[c] = specialCharactersFrequencyDict[specialChar] / totalCount

    vectorOfFrequencies = np.array(vectorOfFrequencies)
    return vectorOfFrequencies


def functionWordsPercentage(inputText):
    functionWords = open(cur_dir_path + "../../writeprintresources/functionWord.txt", "r").readlines()
    functionWords = [f.strip("\n") for f in functionWords]
    # print((functionWords))
    words = text.text_to_word_sequence(inputText, filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"', lower=True, split=" ")
    frequencyOfFunctionWords = []
    for i in range(len(functionWords)):
        functionWord = functionWords[i]
        freq = 0
        for word in words:
            if word == functionWord:
                freq+=1
        frequencyOfFunctionWords.append(freq)
    # functionWordsIntersection = set(words).intersection(set(functionWords))

    return frequencyOfFunctionWords


def frequencyOfPunctuationCharacters(inputText):
    '''
    Calculates the frequency of special characters
    '''

    inputText = str(inputText).lower()  # because its case insensitive
    inputText = inputText.lower().replace(" ", "")
    specialCharacters = open(cur_dir_path + "../../writeprintresources/writeprints_punctuation.txt", "r").readlines()
    specialCharacters = [s.strip("\n") for s in specialCharacters]
    specialCharactersFrequencyDict = {}
    for c in range(0, len(specialCharacters)):
        specialChar = specialCharacters[c]
        specialCharactersFrequencyDict[specialChar] = 0
        for i in str(inputText):
            if specialChar == i:
                specialCharactersFrequencyDict[specialChar] = specialCharactersFrequencyDict[specialChar] + 1

    # making a vector out of the frequencies
    vectorOfFrequencies = [0] * len(specialCharacters)
    totalCount = sum(list(specialCharactersFrequencyDict.values())) + 1
    for c in range(0, len(specialCharacters)):
        specialChar = specialCharacters[c]
        vectorOfFrequencies[c] = specialCharactersFrequencyDict[specialChar] / totalCount

    return vectorOfFrequencies


def misSpellingsPercentage(inputText):
    misspelledWords = open(cur_dir_path + "../../writeprintresources/writeprints_misspellings.txt", "r").readlines()
    misspelledWords = [f.strip("\n") for f in misspelledWords]
    words = text.text_to_word_sequence(inputText, filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"', lower=True, split=" ")
    misspelledWordsIntersection = set(words).intersection(set(misspelledWords))
    return len(misspelledWordsIntersection) / len(list(words))


def legomena(inputText):
    freq = nltk.FreqDist(word for word in inputText.split())
    hapax = [key for key, val in freq.items() if val == 1]
    dis = [key for key, val in freq.items() if val == 2]
    try:
        return list((len(hapax) / len(inputText.split()),len(dis)/ len(inputText.split())))
    except:
        return [0,0]


def posTagFrequency(inputText):
    pos_tags = []
    doc = nlp(str(inputText))
    for token in doc:
        pos_tags.append(str(token.pos_))

    # tagset = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
    tagset = ['ADJ', 'ADP', 'ADV','AUX', 'CONJ','CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'SPACE', 'X']
    tags = [tag for tag in pos_tags]
    return list(tuple(tags.count(tag) / len(tags) for tag in tagset))

def totalWords(inputText):
    words = text.text_to_word_sequence(inputText, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
    return len(words)

def averageWordLength(inputText):
    words = text.text_to_word_sequence(inputText, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
    lengths = []
    for word in words:
        lengths.append(len(word))
    return np.mean(lengths)

def noOfShortWords(inputText):
    words = text.text_to_word_sequence(inputText, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
    shortWords = []
    for word in words:
        if len(word) <= 3:
            shortWords.append(word)
    return len(shortWords)

def calculateFeatures(inputText):


    featureList = []

    ## GROUP 1
    featureList.extend([totalWords(inputText)])
    featureList.extend([averageWordLength(inputText)])
    featureList.extend([noOfShortWords(inputText)])
    ## GROUP 2
    featureList.extend([charactersCount(inputText)])
    featureList.extend([digitsPercentage(inputText)])
    featureList.extend([upperCaseCharactersPercentage(inputText)])
    ## GROUP 3
    featureList.extend(frequencyOfSpecialCharacters(inputText))
    ## GROUP 4
    featureList.extend(frequencyOfLetters(inputText))
    ## GROUP 5
    featureList.extend(frequencyOfDigits(inputText))
    ## GROUP 6
    featureList.extend(mostCommonLetterBigrams(inputText))
    ## GROUP 7
    featureList.extend(mostCommonLetterTrigrams(inputText))
    ## GROUP 8
    featureList.extend(legomena(inputText))
    ## GROUP 9
    featureList.extend(functionWordsPercentage(inputText))
    ## GROUP 10
    featureList.extend(posTagFrequency(inputText))
    ## GROUP 11
    featureList.extend(frequencyOfPunctuationCharacters(inputText))

    return featureList
######################################## TRAINING CLASSIFIER ###########################