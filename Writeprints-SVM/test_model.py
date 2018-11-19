import pickle
import sys
from sklearn.metrics import accuracy_score
def warn(*args, **kwargs):
    pass


import warnings
import json
warnings.warn = warn
import time
import pickle
import os
import sys
import nltk
import spacy
import random
import writeprintsStatic
import jstyloWriteprintsLimited
import gensim
from functools import partial
import numpy as np
from keras.preprocessing import text
from operator import itemgetter
import gensim.models.keyedvectors as word2vec
import math
from multiprocessing.dummy import Pool as ThreadPool
#####################################################

#####################################################

cur_dir_path = os.getcwd() + "/"  # + "writeprints_experiments/"
nlp = spacy.load('en_core_web_sm')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


class Article:
    '''
    Article class that will store all the information about an article
    '''
    originalArticleText = ''
    originalArticleTrailingSpacesWords = []
    originalArticlePOSTags = []
    originalArticleWords = []
    replacementsDict = {}
    fitness = 0
    meteor = 0

    def __init__(self, data):

        self.originalArticleText = ''
        self.originalArticleText = str(data)
        doc = nlp(str(self.originalArticleText))
        self.originalArticleTrailingSpacesWords = []
        self.originalArticlePOSTags = []
        self.originalArticleWords = []
        for token in doc:
            self.originalArticlePOSTags.append(str(token.pos_))
            self.originalArticleWords.append(str(token.text))
            self.originalArticleTrailingSpacesWords.append(str(token.text_with_ws))
        self.replacementsDict = {}


filename = str(sys.argv[1])
test_Instances_filename = str(sys.argv[2])

clf = pickle.load(open( filename, 'rb'))

with open( test_Instances_filename, 'rb') as f:
    test_Instances = pickle.load(f)

X_test = []
y_test = []
total = 0
misclssifed = 0

for i in range(len(test_Instances)):
    filePath, filename, authorLabel, author, featureList, inputText = test_Instances[i]

    print('ARTICLE ====> ', filename, "    Author ====> ", author)

    article = Article(inputText)
    modifiedArticleText = ''.join(str(e) + "" for e in article.originalArticleTrailingSpacesWords)
    # print(modifiedArticleText)
    inital_label_art = clf.predict([writeprintsStatic.calculateFeatures(article.originalArticleText)])[0]
    orignal_article_words = article.originalArticleWords
    initialProb = clf.predict_proba([writeprintsStatic.calculateFeatures(article.originalArticleText)])[0][authorLabel]
    # print (len(test_Instances))
    total+=1
    if authorLabel != inital_label_art:
        print("Classified InCorrectly , Hence Skipping the Article")
        misclssifed+=1

print("Total Test Instances : ", total)
print("Total Misclassified : ", misclssifed)
