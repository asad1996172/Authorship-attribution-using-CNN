import nltk
import numpy as np
import spacy
import pyphen
pyphen.language_fallback('nl_NL_variant1')
from keras.preprocessing import text
dic = pyphen.Pyphen(lang='nl_NL')

############## FEATURES COMPUTATION #####################

def totalWords(inputText):

    # removing punctiation and unifying case
    words = text.text_to_word_sequence(inputText, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
    return len(words)

def complexity(inputText):

    words = text.text_to_word_sequence(inputText, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
    unique_words_count = len(set(words))
    return unique_words_count/len(words)

def sentencesCount(inpuText):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(inpuText)
    return len(sentences)

def averageSentenceLength(inputText):
    totWords = totalWords(inputText)
    totSentences = sentencesCount(inputText)

    return totWords/totSentences

def averageSyllablesInWord(inputText):
    words = text.text_to_word_sequence(inputText, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
    syllableCounts = []
    for word in words:
        syllables = dic.inserted(word)
        syllableCounts.append(len(syllables.split('-')))

    return np.mean(syllableCounts)

def totalComplexWords(inputText):
    words = text.text_to_word_sequence(inputText, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
    complexWords = []
    for word in words:
        syllables = dic.inserted(word)
        if len(syllables.split('-')) >= 3:
            complexWords.append(word)

    return len(complexWords)

def gunningFogReadabilityIndex(inputText):

    totWords = totalWords(inputText)
    totSentences = sentencesCount(inputText)
    totComplexWords = totalComplexWords(inputText)

    return 0.4*((totWords / totSentences) + 100*(totComplexWords / totWords))

def charactersCount(inputText):

    charCount = len(str(inputText))
    return charCount

def totalNumberOfLetters(inputText):

    words = text.text_to_word_sequence(inputText, filters=",.?!\"'`;:-()&$", lower=False, split=" ")
    totalLetters = 0
    for word in words:
        for i in range(len(word)):
            if word[i].isalpha():
                totalLetters+=1

    return totalLetters

def totalSyllables(inputText):
    words = text.text_to_word_sequence(inputText, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
    syllableCount = 0
    for word in words:
        syllables = dic.inserted(word)
        syllableCount+=(len(syllables.split('-')))

    return syllableCount

def fleschReadingEaseScore(inputText):

    totWords = totalWords(inputText)
    totSentences = sentencesCount(inputText)
    totSyllables = totalSyllables(inputText)

    return 206.835 - 1.015*(totWords / totSentences) -84.6*(totSyllables / totWords)


def calculateFeatures(inputText):
    featureList = []

    ## Feature 1
    featureList.append(totalWords(inputText))

    ## Feature 2
    featureList.append(complexity(inputText))

    ## Feature 3
    featureList.append(sentencesCount(inputText))

    ## Feature 4
    featureList.append(averageSentenceLength(inputText))

    ## Feature 5
    featureList.append(averageSyllablesInWord(inputText))

    ## Feature 6
    featureList.append(gunningFogReadabilityIndex(inputText))

    ## Feature 7
    featureList.append(charactersCount(inputText))

    ## Feature 8
    featureList.append(totalNumberOfLetters(inputText))

    ## Feature 9
    featureList.append(fleschReadingEaseScore(inputText))

    return featureList


# dummy = "I believe that The Jungle shows how human beings deep down inside are naturally greedy and self-centered. They only want to look out for themselves and make their lives as good as possible, but never look out for anyone lower then themselves. The Jungle shows this in many ways. The main obvious area is the owners and managers of the great factories.  All they want to do is make their companies and factories as big and great as they can be. They do not care about anyone or anything that gets in their way or anything that gets hurt because of their advancements. According to Upton Sinclair's work, The line of the buildings stood clear-cut and black against the sky; here and there out of the mass rose the great chimneys, with the river of smoke streaming away to the end of the world. Sinclair is trying to paint a picture of the great damage that this pollution is creating. He shows that the owners have no regard for any person or animal that might get hurt. They do not care about the great plumes of smoke billowing out of the factories or the animal feces and urine runoff from the thousands of pens around Packingtown. "
# print(calculate_features(dummy))