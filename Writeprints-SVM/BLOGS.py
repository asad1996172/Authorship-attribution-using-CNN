def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.metrics import accuracy_score
import pickle

import io
import os
import sys
import nltk
import spacy
import gensim
from os import walk
from keras.preprocessing import text
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import writeprintsStatic
import jstyloWriteprintsLimited
from sklearn.svm import SVC

nlp = spacy.load('en_core_web_sm')
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

cur_dir_path = os.getcwd() + "/"  # + "writeprints_experiments/"

classifier = str(sys.argv[2])
featureset = str(sys.argv[3])

if classifier == 'rfc':
    clf = RandomForestClassifier(n_estimators=50,random_state=110)
else:
    clf = SVC(kernel='poly')

authorsNames = []
for (_, directories, _) in walk(cur_dir_path + "../../BlogsAll"):
    authorsNames.extend(directories)

# Parameters to Adjust
authorsToKeep = int(sys.argv[1])


# Classifier Code
if authorsToKeep == 5:
    authorsNames=['1151815','554681','2587254','3010250','3040702']
elif authorsToKeep == 10:
    authorsNames=['1151815','554681','2587254','3010250','3040702','3403444','215223','1046946','1476840','1234212']
X_train, y_train, X_test, y_test = [], [], [], []
authorsNames = list(map(int, authorsNames))
authorsNames = sorted(authorsNames)
authorsNames = list(map(str, authorsNames))
print ("Author Names : ",authorsNames)

authorLabel = 0
train_Instances = []
test_Instances = []

authorsNames = authorsNames[0:authorsToKeep]
for i in range(len(authorsNames)):
    author = authorsNames[i]
    print(author, "====>", i)
    authorFiles = []
    for (_, _, filenames) in walk(cur_dir_path + "../../BlogsAll/" + str(author)):
        authorFiles.extend(filenames)


    count = 0
    authorFiles = sorted(authorFiles)  ###### important to sort for consistency
    for i in range(len(authorFiles)):
        filename = authorFiles[i]
        filePath = cur_dir_path + "../../BlogsAll/" + str(author) + "/" + str(filename)
        inputText = io.open(filePath, "r", errors="ignore").readlines()
        inputText = ''.join(str(e) + "\n" for e in inputText)
        inputText = writeprintsStatic.getCleanText(inputText)
        if featureset == 'static':
            featureList = writeprintsStatic.calculateFeatures(inputText)
        else:
            featureList = jstyloWriteprintsLimited.calculateFeatures(inputText)

        #    FILE SAVING FORMAT
        #   ( FILE PATH , FILE NAME , AUTHOR LABEL , AUTHOR , FEATURE LIST , INITIAL TEXT )

        count = count + 1
        if count <= 80:
            train_Instances.append((filePath,filename,authorLabel,author,featureList,inputText))
            X_train.append(featureList)
            y_train.append(authorLabel)
        else:
            test_Instances.append((filePath, filename, authorLabel, author,featureList,inputText))
            X_test.append(featureList)
            y_test.append(authorLabel)


    authorLabel = authorLabel + 1
# print(test_Instances)
print("TOTAL TEST ARTICLES : ", len(test_Instances))
if not os.path.exists(cur_dir_path + "Pretrained_Files"):
    os.makedirs(cur_dir_path + "Pretrained_Files")

with open(cur_dir_path + "Pretrained_Files/" + str(authorsToKeep) +"_"+classifier+"_"+featureset+ '_test_files_blogs.pkl', 'wb') as f:
    pickle.dump(test_Instances, f)

with open(cur_dir_path + "Pretrained_Files/" + str(authorsToKeep) +"_"+classifier+"_"+featureset+ '_train_files_blogs.pkl', 'wb') as f:
    pickle.dump(train_Instances, f)

if classifier == 'svm' and featureset=="limited":
    from sklearn.preprocessing import MinMaxScaler
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
    X_train = scaling.transform(X_train)
    X_test = scaling.transform(X_test)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Test Accuracy : ", accuracy_score(y_test, y_pred))
filename = cur_dir_path + "Pretrained_Files/" + str(authorsToKeep) +"_"+classifier+"_"+featureset+ '_pretrained_blogs.sav'
pickle.dump(clf, open(filename, 'wb'))