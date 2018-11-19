def warn(*args, **kwargs):
    pass
from sortedcontainers import SortedDict
from collections import Counter
import warnings
import itertools
warnings.warn = warn
import numpy as np
import nltk
import spacy
from keras.preprocessing import text
import re
nlp = spacy.load('en_core_web_sm')

def totalWords(inputText):

    # removing punctiation and unifying case
    words = text.text_to_word_sequence(inputText, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
    return len(words)

def uniqueWordsCount(inputText):

    # removing punctiation and unifying case
    words = text.text_to_word_sequence(inputText, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
    return len(list(set(words))) / totalWords(inputText)

def largerWordsCount(inputText):
    words = text.text_to_word_sequence(inputText, filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"', lower=True, split=" ")
    shortWords = []
    for word in words:
        if len(word) > 4:
            shortWords.append(word)
    return len(shortWords) / totalWords(inputText)

def averageCharactersPerWord(inputText):
    return characterCount(inputText) / totalWords(inputText)

def characterCount(inputText):

    inputText = inputText.lower()
    return len(inputText)


def letters(inputText):
    '''
    Calculates the frequency of letters
    '''

    inputText = str(inputText).lower()  # because its case sensitive
    # inputText = inputText.lower().replace(" ", "")
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
    totalCount = characterCount(inputText)
    for c in range(0, len(characters)):
        char = characters[c]
        vectorOfFrequencies[c] = charsFrequencyDict[char] / totalCount

    return vectorOfFrequencies


def mostCommonLetterBigrams(inputText):
    # to do


    # bigrams = ['th','he','in','er','an','re','nd','at','on','nt','ha','es','st' ,'en','ed','to','it','ou','ea','hi','is','or','ti','as','te','et' ,'ng','of','al','de','se','le','sa','si','ar','ve','ra','ld','ur']
    characters ="abcdefghijklmnopqrstuvwxyz"

    bigrams = []
    tagsets = list(itertools.combinations(characters, 2))
    for tagset in tagsets:
        bigrams.append(''.join(tagset))
    bigramCounts = {}
    for t in bigrams:
        bigramCounts[t] = 0

    totalCount = 0
    words = text.text_to_word_sequence(inputText, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
    for word in words:
        for i in range(0, len(word) - 1):
            bigram = str(word[i:i + 2]).lower()
            if bigram in bigrams:
                bigramCounts[bigram] = bigramCounts[bigram] + 1
                totalCount = totalCount + 1

    bigramCounts = SortedDict(bigramCounts)
    bigramCounts = np.array(bigramCounts.values())
    bigramCounts = sorted(bigramCounts,reverse=True)
    bigramCounts = bigramCounts[:50]
    return np.divide(bigramCounts,characterCount(inputText))


def mostCommonLetterTrigrams(inputText):
    # to do
    # trigrams = ["the", "and", "ing", "her", "hat", "his", "tha", "ere", "for", "ent", "ion", "ter", "was", "you", "ith",
    #             "ver", "all", "wit", "thi", "tio"]
    characters = "abcdefghijklmnopqrstuvwxyz"

    trigrams = []
    tagsets = list(itertools.combinations(characters, 3))
    for tagset in tagsets:
        trigrams.append(''.join(tagset))

    trigramCounts = {}
    for t in trigrams:
        trigramCounts[t] = 0

    totalCount = 0
    words = text.text_to_word_sequence(inputText, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
    for word in words:
        for i in range(0, len(word) - 2):
            trigram = str(word[i:i + 3]).lower()
            if trigram in trigrams:
                trigramCounts[trigram] = trigramCounts[trigram] + 1
                totalCount = totalCount + 1

    trigramCounts = SortedDict(trigramCounts)
    trigramCounts = np.array(trigramCounts.values())
    trigramCounts = sorted(trigramCounts, reverse=True)
    trigramCounts = trigramCounts[:50]
    return np.divide(trigramCounts, characterCount(inputText))



def digits(inputText):

    digits = [0,1,2,3,4,5,6,7,8,9]
    digitsCounts = {}
    for digit in digits:
        digitsCounts[str(digit)] = 0

    alldigits = re.findall('\d', inputText)
    for digit in alldigits:
        digitsCounts[digit] +=1

    digitsCounts = SortedDict(digitsCounts)
    digitsCounts = np.array(digitsCounts.values())
    return np.divide(digitsCounts, characterCount(inputText))

def twoDigitNumbers(inputText):

    digits = []
    for i in range(10,100):
        digits.append(i)
    digitsCounts = {}
    for digit in digits:
        digitsCounts[str(digit)] = 0

    alldigits = re.findall('\d\d', inputText)
    for digit in alldigits:
        try:
            digitsCounts[digit] +=1
        except:
            pass
    digitsCounts = SortedDict(digitsCounts)
    digitsCounts = np.array(digitsCounts.values())
    digitsCounts = sorted(digitsCounts, reverse=True)
    digitsCounts = digitsCounts[:50]
    return np.divide(digitsCounts, characterCount(inputText))

def threeDigitNumbers(inputText):

    digits = []
    for i in range(100,1000):
        digits.append(i)
    digitsCounts = {}
    for digit in digits:
        digitsCounts[str(digit)] = 0

    alldigits = re.findall('\d\d\d', inputText)
    for digit in alldigits:
        try:
            digitsCounts[digit] +=1
        except:
            pass
    digitsCounts = SortedDict(digitsCounts)
    digitsCounts = np.array(digitsCounts.values())
    digitsCounts = sorted(digitsCounts, reverse=True)
    digitsCounts = digitsCounts[:50]
    return np.divide(digitsCounts, characterCount(inputText))

def wordLengths(inputText):

    lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    wordLengthFrequencies = {}
    for l in lengths:
        wordLengthFrequencies[l] = 0

    words = text.text_to_word_sequence(inputText, filters=",.?!\"'`;:-()&$", lower=False, split=" ")
    for w in words:
        wordLength = len(w)
        if wordLength in wordLengthFrequencies:
            wordLengthFrequencies[wordLength] = wordLengthFrequencies[wordLength] + 1

    wordLengthFrequencies = SortedDict(wordLengthFrequencies)
    return wordLengthFrequencies.values()

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
    functionWords = open("writeprintresources/functionWord.txt", "r").readlines()
    functionWords = [f.strip("\n") for f in functionWords]
    # print((functionWords))
    words = text.text_to_word_sequence(inputText, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
    frequencyOfFunctionWords = []
    for i in range(len(functionWords)):
        functionWord = functionWords[i]
        freq = 0
        for word in words:
            if word == functionWord:
                freq+=1
        frequencyOfFunctionWords.append(freq)
    # functionWordsIntersection = set(words).intersection(set(functionWords))

    frequencyOfFunctionWords = np.array(frequencyOfFunctionWords)
    frequencyOfFunctionWords = sorted(frequencyOfFunctionWords, reverse=True)
    frequencyOfFunctionWords = frequencyOfFunctionWords[:50]
    return np.divide(frequencyOfFunctionWords, totalWords(inputText))


def frequencyOfPunctuationCharacters(inputText):
    '''
    Calculates the frequency of special characters
    '''

    inputText = str(inputText).lower()  # because its case insensitive
    inputText = inputText.lower().replace(" ", "")
    specialCharacters = open("writeprintresources/writeprints_punctuation.txt", "r").readlines()
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


def posTagFrequency(inputText):
    pos_tags = []
    doc = nlp(str(inputText))
    for token in doc:
        pos_tags.append(str(token.pos_))

    tagset = ['ADJ', 'ADP', 'ADV','AUX', 'CONJ','CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'SPACE', 'X']
    tags = [tag for tag in pos_tags]
    return list(tuple(tags.count(tag) / len(tags) for tag in tagset))

def posTagBigramFrequency(inputText):
    pos_tags = []
    doc = nlp(str(inputText))
    for token in doc:
        pos_tags.append(str(token.pos_))
    # print(pos_tags)
    # print(doc)
    tagset = ['ADJ', 'ADP', 'ADV','AUX', 'CONJ','CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'SPACE', 'X']
    tagset = list(itertools.combinations(tagset, 2))
    tagset_freq = [0] * len(tagset)
    # print(tagset)
    for i in range(1, len(pos_tags)):
        if (pos_tags[i - 1], pos_tags[i]) in tagset:
            tagset_freq[tagset.index((pos_tags[i - 1], pos_tags[i]))] += 1
            # print (pos_tags[i-1],pos_tags[i])
    # tags = [tag for tag in pos_tags]

    final = [(val / (len(pos_tags))) for val in tagset_freq]
    final = sorted(final,reverse=True)
    return final[:50]


def posTagTrigramFrequency(inputText):
    pos_tags = []
    doc = nlp(str(inputText))
    for token in doc:
        pos_tags.append(str(token.pos_))
    # print(pos_tags)
    # print(doc)
    tagset = ['ADJ', 'ADP', 'ADV','AUX', 'CONJ','CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'SPACE', 'X']
    tagset = list(itertools.combinations(tagset, 3))
    tagset_freq = [0] * len(tagset)
    # print(tagset)
    for i in range(2, len(pos_tags)):
        if (pos_tags[i - 2], pos_tags[i - 1], pos_tags[i]) in tagset:
            tagset_freq[tagset.index((pos_tags[i - 2], pos_tags[i - 1], pos_tags[i]))] += 1
            # print (pos_tags[i-1],pos_tags[i])
    # tags = [tag for tag in pos_tags]

    final = [(val / (len(pos_tags))) for val in tagset_freq]
    final = sorted(final, reverse=True)
    return final[:50]

def unigramWords(inputText):
    words = text.text_to_word_sequence(inputText, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
    values = Counter(words).values()
    values = sorted(values,reverse=True)

    return values[:50]

def bigramWords(inputText):
    words = text.text_to_word_sequence(inputText, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
    bigrams = list(set(list(nltk.bigrams(words))))

    freq_list = [0] * len(bigrams)
    for i in range(1, len(words)):
        if (words[i - 1], words[i]) in bigrams:
            freq_list[bigrams.index((words[i - 1], words[i]))] += 1

    values = [(val / (len(words))) for val in freq_list]
    values = sorted(values, reverse=True)

    return values[:50]

def trigramWords(inputText):
    words = text.text_to_word_sequence(inputText, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
    trigrams = list(set(list(nltk.trigrams(words))))

    freq_list = [0] * len(trigrams)
    for i in range(2, len(words)):
        if (words[i - 2], words[i - 1], words[i]) in trigrams:
            freq_list[trigrams.index((words[i - 2], words[i - 1], words[i]))] += 1

    values = [(val / (len(words))) for val in freq_list]
    values = sorted(values, reverse=True)

    return values[:50]


def misSpellingsPercentage(inputText):

    misspelledWords = open("writeprintresources/writeprints_misspellings.txt", "r").readlines()
    misspelledWords = [f.strip("\n") for f in misspelledWords]
    # print((functionWords))
    words = text.text_to_word_sequence(inputText, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
    frequencyOfFunctionWords = []
    for i in range(len(misspelledWords)):
        functionWord = misspelledWords[i]
        freq = 0
        for word in words:
            if word == functionWord:
                freq += 1
        frequencyOfFunctionWords.append(freq)
    # functionWordsIntersection = set(words).intersection(set(functionWords))

    frequencyOfFunctionWords = np.array(frequencyOfFunctionWords)
    frequencyOfFunctionWords = sorted(frequencyOfFunctionWords, reverse=True)
    frequencyOfFunctionWords = frequencyOfFunctionWords[:50]
    return np.divide(frequencyOfFunctionWords, totalWords(inputText))




def digitsPercentage(inputText):

    charsCount = characterCount(inputText)
    digitsCount = len(re.findall('\d', inputText))
    return digitsCount / charsCount

def lettersPercentage(inputText):

    lettersCount = len(re.findall('[A-Za-z]', inputText))
    charsCount = characterCount(inputText)
    return lettersCount / charsCount

def uppercaseLettersPercentage(inputText):

    lettersCount = len(re.findall('[A-Z]', inputText))
    charsCount = characterCount(inputText)
    return lettersCount / charsCount


def hapaxLegomenon(inputText):

    freq = nltk.FreqDist(word for word in inputText.split())
    hapax = [key for key, val in freq.items() if val == 1]

    try:
        return list(len(hapax) / totalWords(inputText))
    except:
        return [0]

def calculateFeatures(inputText):

    featureList = []

    # Total Words
    featureList.extend([totalWords(inputText)])

    # Total characters
    featureList.extend([characterCount(inputText)])

    # Char per word
    featureList.extend([averageCharactersPerWord(inputText)])

    # Frequencies of letters
    featureList.extend(letters(inputText))

    # Frequencies of top 50 bigrams

    featureList.extend(mostCommonLetterBigrams(inputText))

    # Frequencies of top 50 trigrams

    featureList.extend(mostCommonLetterTrigrams(inputText))

    # Frequencies of digits
    featureList.extend(digits(inputText))

    # Frequencies of two digit numbers

    featureList.extend(twoDigitNumbers(inputText))

    # Frequencies of three digit numbers

    featureList.extend(threeDigitNumbers(inputText))

    # Frequencies of different length words
    featureList.extend(wordLengths(inputText))

    # Frequencies of special characters
    featureList.extend(frequencyOfSpecialCharacters(inputText))

    # Frequencies of function words

    featureList.extend(functionWordsPercentage(inputText))

    # Frequencies of Punctuaiton
    featureList.extend(frequencyOfPunctuationCharacters(inputText))

    # Frequencies of POS tags
    featureList.extend(posTagFrequency(inputText))

    # Frequencies of POS tags bigrams

    featureList.extend(posTagBigramFrequency(inputText))

    # Frequencies of POS tags trigrams

    featureList.extend(posTagTrigramFrequency(inputText))

    # Frequencies of words unigrams

    featureList.extend(unigramWords(inputText))

    # Frequencies of words bigrams

    featureList.extend(bigramWords(inputText))

    # Frequencies of words trigrams
    featureList.extend(trigramWords(inputText))

    # Frequencies of misspelling words
    featureList.extend(misSpellingsPercentage(inputText))

    # Unique word count
    featureList.append(uniqueWordsCount(inputText))

    # digits percentage
    featureList.extend([digitsPercentage(inputText)])

    # letters percentage
    featureList.extend([lettersPercentage(inputText)])

    # upper case letters percentage
    featureList.extend([uppercaseLettersPercentage(inputText)])

    # hapax legomenon
    featureList.extend(hapaxLegomenon(inputText))

    return featureList

# dummy = "I believ@@@@@@e tha1t The3 Jung5le s7how2s4 h2o56w7 h4um7a3n345 beings deep down inside are naturally greedy and self-centered. They only want to look out for themselves and make their lives as good as possible, but never look out for anyone lower then themselves. The Jungle shows this in many ways. The main obvious area is the owners and managers of the great factories.  All they want to do is make their companies and factories as big and great as they can be. They do not care about anyone or anything that gets in their way or anything that gets hurt because of their advancements. According to Upton Sinclair's work, The line of the buildings stood clear-cut and black against the sky; here and there out of the mass rose the great chimneys, with the river of smoke streaming away to the end of the world. Sinclair is trying to paint a picture of the great damage that this pollution is creating. He shows that the owners have no regard for any person or animal that might get hurt. They do not care about the great plumes of smoke billowing out of the factories or the animal feces and urine runoff from the thousands of pens around Packingtown. "
# features = calculateFeatures(dummy)
# print(len(features))
# print(totalWords(dummy))
