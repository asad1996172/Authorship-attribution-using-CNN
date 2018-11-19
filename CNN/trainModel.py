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
from keras.preprocessing.text import Tokenizer,one_hot
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model,Sequential
from keras.layers import Input
from keras.layers import Dense,Reshape,Concatenate
from keras.layers import Flatten,Merge
from keras.layers import Dropout
from keras.layers import Embedding,LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils.np_utils import to_categorical
from keras.layers.merge import concatenate
from gensim import models
import numpy as np
import os
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

cur_dir_path = os.getcwd() + "/"#+ "CNN_appraoch/"

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
            c=1
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


def define_model_CNN_word_word(length, vocab_size,embedding_matrix,word_index,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH):
    embed1 = Sequential()
    embed1.add(Embedding(len(word_index) + 1,
                         EMBEDDING_DIM,
                         weights=[embedding_matrix],
                         input_length=MAX_SEQUENCE_LENGTH,
                         trainable=False))
    embed2 = Sequential()
    embed2.add(Embedding(len(word_index) + 1,
                         EMBEDDING_DIM,
                         weights=[embedding_matrix],
                         input_length=MAX_SEQUENCE_LENGTH,
                         trainable=True))

    model = Sequential()
    model.add(Merge([embed1, embed2], mode='concat', concat_axis=-1))
    # model.add(Reshape((2, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)))

    model.add(Conv1D(64, 5, activation="relu"))
    model.add(MaxPooling1D(MAX_SEQUENCE_LENGTH - 5 + 1))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(int(sys.argv[1]), activation="softmax"))
    # metrics = Metrics()
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    print(model.summary())

    return model



def CNN_word_word(embedding_matrix):
    # load training dataset

    trainLines, trainLabels,testInstances = load_dataset(cur_dir_path + 'modelFiles_blogs/' + str(sys.argv[1]) + '_train.pkl')
    testLines, testLabels,trainInstances = load_dataset(cur_dir_path + 'modelFiles_blogs/' + str(sys.argv[1]) + '_test.pkl')
    # create tokenizer
    tokenizer = create_tokenizer(trainLines)
    sequences = tokenizer.texts_to_sequences(trainLines)
    test_sequences = tokenizer.texts_to_sequences(testLines)
    # calculate max document length
    MAX_SEQUENCE_LENGTH = max_length(trainLines)
    # calculate vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print('Max document length: %d' % MAX_SEQUENCE_LENGTH)
    print('Vocabulary size: %d' % vocab_size)
    # encode data
    # trainX = encode_text(tokenizer, trainLines, MAX_SEQUENCE_LENGTH)
    # print(trainX.shape)

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    x_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # labels = np.asarray(trainLabels)
    labels = to_categorical(trainLabels, num_classes=None)
    y_test = to_categorical(testLabels, num_classes=None)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(0.10 * data.shape[0])  # validation set 10 percent
    print(nb_validation_samples)
    x_train = data
    y_train = labels
    # x_val = data[-nb_validation_samples:]
    # y_val = labels[-nb_validation_samples:]

    print(x_train.shape)
    print(y_train.shape)
    # print(x_val.shape)
    # print(y_val.shape)


    EMBEDDING_DIM = 300
    word_index = tokenizer.word_index
    print("loading GLOVE model")

    print("Filling non existing words")
    embedding_matrix = fill_in_missing_words_with_zeros(embedding_matrix, tokenizer.word_index, EMBEDDING_DIM)
    # embedding_matrix = fill_in_missing_words_with_zeros({}, tokenizer.word_index, EMBEDDING_DIM)

    # define model
    model = define_model_CNN_word_word(MAX_SEQUENCE_LENGTH, vocab_size,embedding_matrix,word_index,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH)

    model.fit([x_train,x_train], y_train,
              epochs=15, batch_size=50)
    # save the model
    # model.model.save()

    # serialize model to JSON
    model_json = model.to_json()
    with open(cur_dir_path + 'savedModelFiles_blogs/' + str(sys.argv[1]) + '_CNN_word_word_model.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(cur_dir_path + 'savedModelFiles_blogs/' + str(sys.argv[1]) + '_CNN_word_word_model.h5')
    print("Saved model to disk")

    # load json and create model
    json_file = open(cur_dir_path + 'savedModelFiles_blogs/' +str(sys.argv[1]) + '_CNN_word_word_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(cur_dir_path + 'savedModelFiles_blogs/' +str(sys.argv[1]) + '_CNN_word_word_model.h5')
    print("Loaded model from disk")
    loaded_model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    loss, acc = loaded_model.evaluate([x_test,x_test],y_test, verbose=0)
    print('Test Accuracy: %f' % (acc*100))

    loss, acc = model.evaluate([x_test,x_test],y_test, verbose=0)
    print('Test Accuracy: %f' % (acc*100))

    return acc,model

embedding_matrix = load_model('modelFiles_blogs/glove.840B.300d.txt')

cnn_word_word,model = CNN_word_word(embedding_matrix)

print("CNN_word_word Accuracy for" , str(sys.argv[1]) , "Authors :" ,cnn_word_word )
