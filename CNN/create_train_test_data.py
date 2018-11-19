# from os import walk
# from pickle import dump
# from keras.preprocessing import text
# import os
# import sys
# import io
#
# def getCleanText(inputText):
#     # cleanText = text.text_to_word_sequence(inputText,filters='!#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"\' ', lower=False, split=" ")
#     cleanText = text.text_to_word_sequence(inputText, filters='', lower=False, split=" ")
#
#     cleanText = ''.join(str(e) + " " for e in cleanText)
#     return cleanText
#
#
# # save a dataset to file
# def save_dataset(dataset, filename):
#     dump(dataset, open(filename, 'wb'))
#     print('Saved: %s' % filename)
#
# def create_data(dataset):
#     cur_dir_path = os.getcwd() + "/"
#     authorsNames = []
#     for (_, directories, _) in walk(cur_dir_path +"../../" + dataset):
#         authorsNames.extend(directories)
#
#
#     authorsToKeep = int(sys.argv[1])	#<====================############################
#
#
#     X_train, y_train, X_test, y_test = [], [], [], []
#     # Classifier Code
#     if authorsToKeep == 5:
#         authorsNames = ['h', 'qq', 'pp', 'm', 'y']
#     elif authorsToKeep == 10:
#         authorsNames = ['h', 'qq', 'pp', 'm', 'y', 'b', 'c', 'ss', 'o', 'w']
#     # authorsNames = list(map(int, authorsNames))
#     authorsNames = sorted(authorsNames)
#     # authorsNames = list(map(str, authorsNames))
#     modelAuthors = []
#     authorLabel = 0
#     testArticles = []
#     trainArticles = []
#
#     train_Instances = []
#     test_Instances = []
#
#     labels_dictionry = {}
#
#     authorsNames = authorsNames[0:authorsToKeep]
#     for i in range(0, authorsToKeep):
#         author = authorsNames[i]
#         print("Author: " ,author,"====>",i)
#         authorFiles = []
#         for (_, _, filenames) in walk(cur_dir_path + "../../"+dataset + "/" + str(author)):
#             authorFiles.extend(filenames)
#
#             # if len(authorFiles) <= 12:
#         #	continue
#         labels_dictionry[author] = authorLabel
#         count = 0
#         authorFiles = sorted(authorFiles)
#         for filename in authorFiles:
#             filePath = cur_dir_path + "../../" + dataset + "/" + str(author) + "/" + str(filename)
#             inputText = io.open(filePath, "r", errors="ignore").readlines()
#             inputText = ''.join(str(e) + "\n" for e in inputText)
#             inputText = getCleanText(inputText)
#             print(authorLabel,count)
#
#             count = count + 1
#             if count <= 12:
#                 train_Instances.append((filePath, filename, authorLabel, author, inputText))
#                 trainArticles.append(filename)
#                 X_train.append(inputText)
#                 y_train.append(authorLabel)
#             else:
#                 test_Instances.append((filePath, filename, authorLabel, author, inputText))
#                 testArticles.append(filePath)
#                 X_test.append(inputText)
#                 y_test.append(authorLabel)
#
#         modelAuthors.append(author)
#         authorLabel = authorLabel + 1
#
#     print(len(testArticles))
#     return trainArticles,X_train,y_train,testArticles,X_test,y_test,train_Instances,test_Instances
#
#
# # load all training reviews
# trainArticles,X_train,y_train,testArticles,X_test,y_test,train_Instances,test_Instances = create_data("amt")
#
# if not os.path.exists("modelFiles_amt"):
#     os.makedirs("modelFiles_amt")
#
# save_dataset([X_train, y_train,train_Instances], 'modelFiles_amt/' + str(sys.argv[1]) + '_train.pkl')
# save_dataset([X_test, y_test,test_Instances], 'modelFiles_amt/' + str(sys.argv[1]) + '_test.pkl')
# dump(testArticles, open('modelFiles_amt/' + str(sys.argv[1]) + '_testArticlePaths.pkl', 'wb'))
# dump(trainArticles, open('modelFiles_amt/' + str(sys.argv[1]) + '_trainArticlePaths.pkl', 'wb'))



# for Blogs
from os import walk
from pickle import dump
from keras.preprocessing import text
import os
import sys
import io

def getCleanText(inputText):
    # cleanText = text.text_to_word_sequence(inputText,filters='!#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"\' ', lower=False, split=" ")
    cleanText = text.text_to_word_sequence(inputText, filters='', lower=False, split=" ")

    cleanText = ''.join(str(e) + " " for e in cleanText)
    return cleanText


# save a dataset to file
def save_dataset(dataset, filename):
    dump(dataset, open(filename, 'wb'))
    print('Saved: %s' % filename)

def create_data(dataset):
    cur_dir_path = os.getcwd() + "/"
    authorsNames = []
    for (_, directories, _) in walk(cur_dir_path +"../../" + dataset):
        authorsNames.extend(directories)


    authorsToKeep = int(sys.argv[1])	#<====================############################


    X_train, y_train, X_test, y_test = [], [], [], []
    # Classifier Code
    if authorsToKeep == 5:
        authorsNames = ['1151815', '554681', '2587254', '3010250', '3040702']
    elif authorsToKeep == 10:
        authorsNames = ['1151815', '554681', '2587254', '3010250', '3040702', '3403444', '215223', '1046946', '1476840',
                        '1234212']
    # authorsNames = list(map(int, authorsNames))
    authorsNames = sorted(authorsNames)
    # authorsNames = list(map(str, authorsNames))
    modelAuthors = []
    authorLabel = 0
    testArticles = []
    trainArticles = []

    train_Instances = []
    test_Instances = []

    labels_dictionry = {}

    authorsNames = authorsNames[0:authorsToKeep]
    for i in range(0, authorsToKeep):
        author = authorsNames[i]
        print("Author: " ,author,"====>",i)
        authorFiles = []
        for (_, _, filenames) in walk(cur_dir_path + "../../"+dataset + "/" + str(author)):
            authorFiles.extend(filenames)

            # if len(authorFiles) <= 12:
        #	continue
        labels_dictionry[author] = authorLabel
        count = 0
        authorFiles = sorted(authorFiles)
        for filename in authorFiles:
            filePath = cur_dir_path + "../../" + dataset + "/" + str(author) + "/" + str(filename)
            inputText = io.open(filePath, "r", errors="ignore").readlines()
            inputText = ''.join(str(e) + "\n" for e in inputText)
            inputText = getCleanText(inputText)
            print(authorLabel,count)

            count = count + 1
            if count <= 80:
                train_Instances.append((filePath, filename, authorLabel, author, inputText))
                trainArticles.append(filename)
                X_train.append(inputText)
                y_train.append(authorLabel)
            else:
                test_Instances.append((filePath, filename, authorLabel, author, inputText))
                testArticles.append(filePath)
                X_test.append(inputText)
                y_test.append(authorLabel)

        modelAuthors.append(author)
        authorLabel = authorLabel + 1

    print(len(testArticles))
    return trainArticles,X_train,y_train,testArticles,X_test,y_test,train_Instances,test_Instances


# load all training reviews
trainArticles,X_train,y_train,testArticles,X_test,y_test,train_Instances,test_Instances = create_data("BlogsAll")

if not os.path.exists("modelFiles"):
    os.makedirs("modelFiles")

save_dataset([X_train, y_train,train_Instances], 'modelFiles/' + str(sys.argv[1]) + '_train.pkl')
save_dataset([X_test, y_test,test_Instances], 'modelFiles/' + str(sys.argv[1]) + '_test.pkl')
dump(testArticles, open('modelFiles/' + str(sys.argv[1]) + '_testArticlePaths.pkl', 'wb'))
dump(trainArticles, open('modelFiles/' + str(sys.argv[1]) + '_trainArticlePaths.pkl', 'wb'))