def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
import io
import os
import sys
import nltk
import spacy
import gensim
from os import walk
from keras.preprocessing import text
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import writeprintsStatic
import jstyloWriteprintsLimited
nlp = spacy.load('en_core_web_sm')
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

cur_dir_path = os.getcwd() + "/"  # + "writeprints_experiments/"

classifier = str(sys.argv[2])
featureset = str(sys.argv[3])

if classifier == 'rfc':
    clf = RandomForestClassifier(n_estimators=50,random_state=120)
else:
    clf = SVC(kernel='poly')

authorsNames = []
for (_, directories, _) in walk(cur_dir_path + "../../amt"):
    authorsNames.extend(directories)

# Parameters to Adjust
authorsToKeep = int(sys.argv[1])


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
        filePath = cur_dir_path + "../../amt/" + str(author) + "/" + str(filename)
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
        if count <= 12:
            train_Instances.append((filePath,filename,authorLabel,author,featureList,inputText))
            X_train.append(featureList)
            print(len(featureList))
            y_train.append(authorLabel)
        else:
            test_Instances.append((filePath, filename, authorLabel, author,featureList,inputText))
            X_test.append(featureList)
            y_test.append(authorLabel)


    authorLabel = authorLabel + 1


print(test_Instances)
print("TOTAL TEST ARTICLES : ", len(test_Instances))
if not os.path.exists(cur_dir_path + "Pretrained_Files"):
    os.makedirs(cur_dir_path + "Pretrained_Files")

with open(cur_dir_path + "Pretrained_Files/" + str(authorsToKeep)+"_"+classifier+"_"+featureset+ '_test_files_amt.pkl', 'wb') as f:
    pickle.dump(test_Instances, f)

with open(cur_dir_path + "Pretrained_Files/" + str(authorsToKeep) +"_"+classifier+"_"+featureset+ '_train_files_amt.pkl', 'wb') as f:
    pickle.dump(train_Instances, f)


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Test Accuracy : ", accuracy_score(y_test, y_pred))
filename = cur_dir_path + "Pretrained_Files/" + str(authorsToKeep) +"_"+ classifier+"_"+featureset+'_pretrained_amt.sav'
pickle.dump(clf, open(filename, 'wb'))