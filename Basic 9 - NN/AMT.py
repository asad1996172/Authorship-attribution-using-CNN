def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
import pickle
import io
import os
import sys
import nltk
from sklearn.neural_network import MLPClassifier
import spacy
import gensim
from os import walk
from keras.preprocessing import text
from sklearn.ensemble import RandomForestClassifier
from readability import readability as extract_features

nlp = spacy.load('en_core_web_sm')
import matplotlib.pyplot as plt

nlp = spacy.load('en_core_web_sm')
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

cur_dir_path = os.getcwd() + "/"  # + "writeprints_experiments/"
wordVectorModel = gensim.models.Word2Vec.load(cur_dir_path + "../../Word Embeddings/gensimWord2Vec.bin")
# Parameters to Adjust
authorsToKeep = int(sys.argv[1])


############## FEATURES COMPUTATION #####################
def getCleanText(inputText):
    # cleanText = text.text_to_word_sequence(inputText,filters='!#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"\' ', lower=False, split=" ")
    cleanText = text.text_to_word_sequence(inputText, filters='', lower=False, split=" ")

    cleanText = ''.join(str(e) + " " for e in cleanText)
    return cleanText


######################################## TRAINED CLASSIFIER ###########################
# (number of features + number of classes ) / 2
if authorsToKeep == 5:
    clf = MLPClassifier(hidden_layer_sizes=(5, 7), activation='relu', learning_rate_init=0.009,random_state=100)
else:
    clf = MLPClassifier(hidden_layer_sizes=(5, 9), activation='relu', learning_rate_init=0.009,random_state=100)

######################################## TRAINING CLASSIFIER ###########################


authorsNames = []
for (_, directories, _) in walk(cur_dir_path + "../../amt"):
    authorsNames.extend(directories)


# Classifier Code
if authorsToKeep == 5:
    authorsNames=['h','qq','pp','m','y']
elif authorsToKeep == 10:
    authorsNames=['h','qq','pp','m','y','b','c','ss','o','w']
X_train, y_train, X_test, y_test = [], [], [], []
authorsNames = sorted(authorsNames)
print ("Author Names : ",authorsNames)

authorLabel = 0
train_Instances = []
test_Instances = []

authorsNames = authorsNames[0:authorsToKeep]
for i in range(len(authorsNames)):
    author = authorsNames[i]
    print(author, "====>", i)
    authorFiles = []
    for (_, _, filenames) in walk(cur_dir_path + "../../amt/" + str(author)):
        authorFiles.extend(filenames)


    count = 0
    authorFiles = sorted(authorFiles)  ###### important to sort for consistency
    for i in range(len(authorFiles)):

        filename = authorFiles[i]
        print(filename)
        filePath = cur_dir_path + "../../amt/" + str(author) + "/" + str(filename)
        inputText = io.open(filePath, "r", errors="ignore").readlines()
        inputText = ''.join(str(e) + "\n" for e in inputText)
        inputText = getCleanText(inputText)
        featureList = extract_features.calculateFeatures(inputText)
        print(len(featureList))
        #    FILE SAVING FORMAT
        #   ( FILE PATH , FILE NAME , AUTHOR LABEL , AUTHOR , FEATURE LIST , INITIAL TEXT )

        count = count + 1
        if count <= 12:
            train_Instances.append((filePath,filename,authorLabel,author,featureList,inputText))
            X_train.append(featureList)
            y_train.append(authorLabel)
        else:
            test_Instances.append((filePath, filename, authorLabel, author,featureList,inputText))
            X_test.append(featureList)
            y_test.append(authorLabel)


    authorLabel = authorLabel + 1

global scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf.fit(X_train, y_train)
print(test_Instances)
print("TOTAL TEST ARTICLES : ", len(test_Instances))
if not os.path.exists(cur_dir_path + "Pretrained_Files"):
    os.makedirs(cur_dir_path + "Pretrained_Files")

with open(cur_dir_path + "Pretrained_Files/" + str(authorsToKeep) + '_test_files_amt.pkl', 'wb') as f:
    pickle.dump(test_Instances, f)

with open(cur_dir_path + "Pretrained_Files/" + str(authorsToKeep) + '_train_files_amt.pkl', 'wb') as f:
    pickle.dump(train_Instances, f)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Test Accuracy : ", accuracy_score(y_test, y_pred))
filename = cur_dir_path + "Pretrained_Files/" + str(authorsToKeep) + '_random_forrest_pretrained_amt.sav'
pickle.dump(clf, open(filename, 'wb'))