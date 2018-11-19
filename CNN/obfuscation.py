import warnings
import json
from operator import itemgetter
import _pickle as cPickle
import time

import io

import spacy
import random
import gensim
from os import walk

from keras.preprocessing import text

global clf
import gensim
import spacy
import sys
from keras.models import model_from_json
from pickle import load
import io
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Dense, Reshape, Concatenate
from keras.layers import Flatten, Merge
from keras.layers import Dropout
from keras.layers import Embedding, LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils.np_utils import to_categorical
from keras.layers.merge import concatenate
from gensim import models
import numpy as np
import os
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

cur_dir_path = os.getcwd()  + "/" #+ "CNN_appraoch/"


# load a clean dataset
def load_dataset(filename):
    return load(open(filename, 'rb'))


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def create_tokenizer_char(lines):
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(lines)
    return tokenizer


def fill_in_missing_words_with_zeros(embeddings_index, word_index, EMBEDDING_DIM):
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


# calculate the maximum document length
def max_length(lines):
    return max([len(s.split()) for s in lines])


def load_model(filename):
    embeddings_index = {}
    f = open(cur_dir_path + filename)
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            if len(coefs) == 300:
                embeddings_index[word] = coefs
        except:
            # print(values)
            c = 1
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


# encode a list of lines
def encode_text(tokenizer, lines, length):
    # integer encode
    encoded = tokenizer.texts_to_sequences(lines)
    # pad encoded sequences
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded




global clf

# load json and create model
json_file = open(cur_dir_path + "savedmodelFiles_amt/"+str(sys.argv[1]) + '_CNN_word_word_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
clf = model_from_json(loaded_model_json)
# load weights into new model
clf.load_weights(cur_dir_path + "savedmodelFiles_amt/"+ str(sys.argv[1]) + '_CNN_word_word_model.h5')
print("Loaded model from disk")
clf.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

def load_dataset(filename):
    return load(open(filename, 'rb'))



# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def fill_in_missing_words_with_zeros(embeddings_index, word_index, EMBEDDING_DIM):
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

# calculate the maximum document length
def max_length(lines):
    return max([len(s.split()) for s in lines])


nlp = spacy.load('en_core_web_sm')
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
random.seed(sys.argv[1])

# cur_dir_path = os.getcwd() + "/"


trainLines, trainLabels,train_Instances = load_dataset(cur_dir_path + 'modelFiles_amt/' + str(sys.argv[1]) + '_train.pkl')
testLines, testLabels,test_Instances = load_dataset(cur_dir_path + 'modelFiles_amt/' + str(sys.argv[1]) + '_test.pkl')
# create tokenizer
tokenizer = create_tokenizer(trainLines)
MAX_SEQUENCE_LENGTH = max_length(trainLines)
vocab_size = len(tokenizer.word_index) + 1

test_sequences = tokenizer.texts_to_sequences(testLines)

print('Max document length: %d' % MAX_SEQUENCE_LENGTH)
print('Vocabulary size: %d' % vocab_size)

x_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

# labels = np.asarray(trainLabels)
labels = to_categorical(trainLabels, num_classes=None)
y_test = to_categorical(testLabels, num_classes=None)

loss, acc = clf.evaluate([x_test,x_test],y_test, verbose=0)
print('Test Accuracy: %f' % (acc*100))

# create tokenizer

print('Max document length: %d' % MAX_SEQUENCE_LENGTH)
print('Vocabulary size: %d' % vocab_size)
wordVectorModel = gensim.models.Word2Vec.load(cur_dir_path + "../../Word Embeddings/gensimWord2Vec.bin")

def getPredictionProbability(inputText):
    sequences = tokenizer.texts_to_sequences([inputText])
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    classes = clf.predict_classes([np.array(data),np.array(data)])
    # classes = np.argmax(probabilities, axis=1)
    prob = clf.predict_proba([np.array(data),np.array(data)])[0][classes]
    return classes,prob[0]

class Article:
    '''
    Article class that will store all the information about an article
    '''
    originalArticleText = ''
    originalFeaturesText = []
    originalArticlePOSTags = []
    originalArticleWords = []
    replacementsDict = {}
    neighborsK = 2
    fitness = 0

    def __init__(self, data):
        self.originalArticleText = ''
        self.originalArticleText = str(data)
        doc = nlp(str(self.originalArticleText))
        self.originalArticlePOSTags = []
        self.originalArticleWords = []
        for token in doc:
            self.originalArticlePOSTags.append(str(token.pos_))
            self.originalArticleWords.append(str(token.text))
        self.replacementsDict = {}



# wordBigramVectorModel = gensim.models.Word2Vec.load("Word Embeddings/gensimWord2Vec_bigram_2.bin")


############## FEATURES COMPUTATION #####################
def getCleanText(inputText):
    # cleanText = text.text_to_word_sequence(inputText,filters='!#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"\' ', lower=False, split=" ")
    cleanText = text.text_to_word_sequence(inputText, filters='', lower=False, split=" ")

    cleanText = ''.join(str(e) + " " for e in cleanText)
    return cleanText


######################################## TRAINED CLASSIFIER ###########################


def computeNeighbors(word, neighborsCount=1):
    # compute top k neighbors of a word using word embeddings

    word = str(word).lower()
    try:
        neighbors = list(wordVectorModel.similar_by_word(word, neighborsCount))
    except:
        return -1

    distances = [neighbor[1] for neighbor in neighbors]
    distancesList = [neighbor[0] for neighbor in neighbors if neighbor[1] > 0.75]
    if len(distancesList) == 0:
        return -1

    return distancesList[:neighborsCount]


def computeBigramNeighbors(bigram, neighborsCount=1):
    word = str(bigram).lower()
    try:
        neighbors = list(wordBigramVectorModel.similar_by_word(word, neighborsCount))
    except:
        return -1

    distances = [neighbor[1] for neighbor in neighbors]
    distancesList = [neighbor[0] for neighbor in neighbors if neighbor[1] > 0.80]
    if len(distancesList) == 0:
        return -1

    return distancesList[:neighborsCount]


def makeReplacement(article, n):
    replacementsCount = random.randrange(2, n)
    genotypes = []
    modifiedArticle = list(article.originalArticleWords)
    modifiedArticleText = ''.join(str(e) + " " for e in modifiedArticle)
    modifiedArticleObject = Article(modifiedArticleText)
    modifiedArticleObject.replacementsDict.update(article.replacementsDict)

    for i in range(replacementsCount):
        # select a random pos tag and a random word to replace
        randomPosition = random.randrange(0, len(article.originalArticlePOSTags) - 1)
        randomPOS = article.originalArticlePOSTags[randomPosition]
        randomWord = ''

        while (
                                    randomPOS != 'NOUN' and randomPOS != 'VERB' and randomPOS != 'ADJ' and randomPOS != 'ADV' and randomPOS != 'ADP' and randomPOS != 'PUNCT' and randomPOS != 'PART') or randomWord == '':
            randomPosition = random.randrange(0, len(article.originalArticlePOSTags) - 1)
            randomPOS = article.originalArticlePOSTags[randomPosition]
            randomWord = article.originalArticleWords[randomPosition]
            neighbors = computeNeighbors(randomWord, 1)
            if neighbors == -1:
                randomWord = ''

        neighbors = computeNeighbors(randomWord, 1)
        if neighbors == -1:
            continue

            # replace word with neighbor
        neig = neighbors[0]
        previousWord = str(modifiedArticleObject.originalArticleWords[randomPosition])
        modifiedArticleObject.originalArticleWords[randomPosition] = neig
        modifiedArticleObject.replacementsDict[previousWord] = neig
        # print('-- Number of replacements in the current article',len(modifiedArticleObject.replacementsDict), len(article.replacementsDict))

    modifiedArticleObject.originalArticleText = ''.join(
        str(e) + " " for e in modifiedArticleObject.originalArticleWords)
    return modifiedArticleObject


def calculateFitness(inputText, replacementDictionary, authorId):
    # hyperparameters
    a = 0.75
    b = 0.25

    # prob
    label,probability = getPredictionProbability(inputText)

    distance = []

    replacements = len(replacementDictionary)
    for word in replacementDictionary:
        neighbor = replacementDictionary[word]
        try:
            sim = wordVectorModel.similarity(word, neighbor)
            dis = 1 - sim
            distance.append(dis)
        except:
            z = 1

    if len(distance) > 0:
        distance = np.mean(distance)
    else:
        distance = 0

    score = (a * probability) + (b * distance)
    score = 1 / score

    try:
        return score, probability, distance, replacements, label
    except:
        return 0, 0, 0, 0,0



# print(allArticles)
replacementsValue = 5
generation = 5
topK = 5
iterations = 0
authorsToKeep = int(sys.argv[1])
total_mislabelled = 0
total_test_cases = 0
total_correctly_classified = 0
for (articlePath,singleArticle,authorId,author,articleData) in test_Instances:

    articleData = getCleanText(articleData)
    inital_label_art, initialProb = getPredictionProbability(articleData)
    article = Article(articleData)

    if not os.path.exists(cur_dir_path + "ResultsBLOGS" + str(authorsToKeep)):
        os.makedirs(cur_dir_path + "ResultsBLOGS" + str(authorsToKeep))
    outputResults = open(cur_dir_path + "ResultsBLOGS" + str(authorsToKeep) + "/" + str(authorId) + "-" + str(author), "w")
    # print("initial_features:", scaler.transform([extract_features.calculateFeatures(''.join(str(e) + " " for e in article.originalArticleWords))]))
    modifiedArticle = list(article.originalArticleWords)
    modifiedArticleText = ''.join(str(e) + " " for e in modifiedArticle)
    modifiedArticleText = " ".join(modifiedArticleText.split())
    inital_label_art , initialProb = getPredictionProbability(modifiedArticleText)

    print(initialProb)
    print('*** INITIAL PROBABILITY', round(initialProb, 4))
    outputResults.write(
        "Iteration, Generation, Topk, Replacement Value" + str(iterations) + "," + str(generation) + "," + str(
            topK) + "," + str(replacementsValue) + "\n")
    outputResults.write("Initial Probability of detection :: " + str(initialProb) + "\n")
    outputResults.write("Orignal Label :: " + str(authorId) + "\n")
    # inital_label_art = clf.predict(scaler.transform([extract_features.calculateFeatures(modifiedArticleText)]))[0]
    outputResults.write("Inital Predicted Label :: " + str(inital_label_art) + "\n\n\n")

    base_interim_file_save = cur_dir_path + str(authorsToKeep) + "_Blogs_Obfuscated_Articles_GA_Data"
    if not os.path.exists(base_interim_file_save + "/" + singleArticle):
        os.makedirs(base_interim_file_save + "/" + singleArticle)

    f = open(base_interim_file_save + "/" + singleArticle + "/Orignal_Article.txt", "w")
    modifiedArticle = list(article.originalArticleWords)
    modifiedArticleText = ''.join(str(e) + " " for e in modifiedArticle)


    f.write(modifiedArticleText)
    f.close()

    # modified articles list
    modifiedArticles = []

    for i in range(0, generation):
        mArticle = makeReplacement(article, replacementsValue)
        modifiedArticles.append(mArticle)

    topArticles = []
    for m in modifiedArticles:
        modifiedArticle = list(m.originalArticleWords)
        modifiedArticleText = ''.join(str(e) + " " for e in modifiedArticle)
        modifiedArticleText = " ".join(modifiedArticleText.split())
        fitness, prob, distance, numReplacements,label = calculateFitness(modifiedArticleText, m.replacementsDict, authorId)
        m.fitness = fitness
        # print(0, fitness, distance, prob, numReplacements,label)
        outputResults.write(
            str(0) + "," + str(fitness) + "," + str(distance) + "," + str(prob) + "," + str(numReplacements)+ "," + str(label) + "\n")
        topArticles.append([fitness, distance, prob, numReplacements, m])

    topArticles = list(reversed(sorted(topArticles, key=itemgetter(0))))
    topArticles = topArticles[:topK]
    if not os.path.exists(cur_dir_path + base_interim_file_save + "/" + singleArticle + "/Iter_" + str(0)):
        os.makedirs(cur_dir_path + base_interim_file_save + "/" + singleArticle + "/Iter_" + str(0))

    toparticle = topArticles[0]
    toparticle = toparticle[4]
    f = open(cur_dir_path + base_interim_file_save + "/" + singleArticle + "/Iter_" + str(0) + "/" + "Best Article.txt",
             "w")
    modifiedArticle = list(toparticle.originalArticleWords)
    modifiedArticleText = ''.join(str(e) + " " for e in modifiedArticle)
    modifiedArticleText = " ".join(modifiedArticleText.split())
    f.write(modifiedArticleText)
    # f.write("\n\nChanges Made in Previous Iteration Article\n")
    # f.write(json.dumps(toparticle.replacementsDict))
    # f.write('\n')
    # f.write("Fitness Score\n")
    # f.write(str(m.fitness))
    f.close()

    for i in range(len(topArticles)):
        artilces = topArticles[i]
        toparticle = artilces[4]
        f = open(cur_dir_path + base_interim_file_save + "/" + singleArticle + "/Iter_" + str(0) + "/" + str(i) + ".txt",
                 "w")
        modifiedArticle = list(toparticle.originalArticleWords)
        modifiedArticleText = ''.join(str(e) + " " for e in modifiedArticle)
        modifiedArticleText = " ".join(modifiedArticleText.split())
        f.write(modifiedArticleText)
        f.write("\n\nChanges Made in Previous Iteration Article\n")
        f.write(json.dumps(toparticle.replacementsDict))
        f.write('\n')
        f.write("Fitness Score\n")
        f.write(str(m.fitness))
        f.close()


    # print(topArticles[0][:4])
    modifiedArticles[:] = []
    last_label = 0
    for x in range(1, iterations):
        print("Iteration =====> ", x)

        for t in topArticles:
            for i in range(0, generation):
                tArticle = makeReplacement(t[-1], replacementsValue)
                modifiedArticles.append(tArticle)

        for m in modifiedArticles:
            modifiedArticle = list(m.originalArticleWords)
            modifiedArticleText = ''.join(str(e) + " " for e in modifiedArticle)
            modifiedArticleText = " ".join(modifiedArticleText.split())
            fitness, prob, distance, numReplacements, label = calculateFitness(modifiedArticleText, m.replacementsDict,
                                                                        authorId)
            m.fitness = fitness
            # print(x, fitness, distance, prob, numReplacements,label)
            outputResults.write(
                str(x) + "," + str(fitness) + "," + str(distance) + "," + str(prob) + "," + str(numReplacements)+ "," + str(label) + "\n")
            topArticles.append([fitness, distance, prob, numReplacements, m])

        last_label = label
        topArticles = list(reversed(sorted(topArticles, key=itemgetter(0))))
        topArticles = topArticles[:topK]

        if not os.path.exists(cur_dir_path + base_interim_file_save + "/" + singleArticle + "/Iter_" + str(x)):
            os.makedirs(cur_dir_path + base_interim_file_save + "/" + singleArticle + "/Iter_" + str(x))

        toparticle = topArticles[0]
        toparticle = toparticle[4]
        f = open(cur_dir_path + base_interim_file_save + "/" + singleArticle + "/Iter_" + str(x) + "/" + "Best Article.txt",
                 "w")
        modifiedArticle = list(toparticle.originalArticleWords)
        modifiedArticleText = ''.join(str(e) + " " for e in modifiedArticle)
        modifiedArticleText = " ".join(modifiedArticleText.split())
        f.write(modifiedArticleText)
        # f.write("\n\nChanges Made in Previous Iteration Article\n")
        # f.write(json.dumps(toparticle.replacementsDict))
        # f.write('\n')
        # f.write("Fitness Score\n")
        # f.write(str(m.fitness))
        f.close()

        for i in range(len(topArticles)):
            artilces = topArticles[i]
            toparticle = artilces[4]
            f = open(cur_dir_path + base_interim_file_save + "/" + singleArticle + "/Iter_" + str(x) + "/" + str(i) + ".txt",
                     "w")
            modifiedArticle = list(toparticle.originalArticleWords)
            modifiedArticleText = ''.join(str(e) + " " for e in modifiedArticle)
            modifiedArticleText = " ".join(modifiedArticleText.split())
            f.write(modifiedArticleText)
            f.write("\n\nChanges Made in Previous Iteration Article\n")
            f.write(json.dumps(toparticle.replacementsDict))
            f.write('\n')
            f.write("Fitness Score\n")
            f.write(str(m.fitness))
            f.close()

        # print(topArticles[0][:4])
        modifiedArticles[:] = []

    total_test_cases+=1
    if authorId == inital_label_art:
        total_correctly_classified+=1
        if inital_label_art != last_label:
            total_mislabelled +=1
    outputResults.close()

print("Total Test Instances : ", total_test_cases)
print("Total Correctly Labelled : ", total_correctly_classified)
print("Total Misclassified via Obfuscation : ", total_mislabelled)