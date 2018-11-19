# Authorship Attribution using CNNs
Given a set of documents by certain authors, correctly identify their authors using CNNs.

# Project Idea
I am going to be working on text classification using Convolutional Neural Networks (CNNs). Main Idea of my project is to classify blogs, given a set of blogs by a certain author correctly classify it. I am also going to compare it to some state of the art Machine Learning methods for authorship attribution.

# Dataset
The dataset to be used in this project is called Blogger dataset. The collected posts are from 19,320 bloggers gathered from blogger.com in August 2004. I am downloading corpus from (http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm). According to this source "The corpus incorporates a total of 681,288 posts and over 140 million words - or approximately 35 posts and 7250 words per person.  

Each blog is presented as a separate file, the name of which indicates a blogger id# and the blogger’s self-provided gender, age, industry and astrological sign. (All are labeled for gender and age but for many, industry and/or sign is marked as unknown.)

All bloggers included in the corpus fall into one of three age groups:

        8240 "10s" blogs (ages 13-17)

        8086 "20s" blogs(ages 23-27)

        2994 "30s" blogs (ages 33-47)

For each age group there are an equal number of male and female bloggers.   

Each blog in the corpus includes at least 200 occurrences of common English words. All formatting has been stripped with two exceptions. Individual posts within a single blogger are separated by the date of the following post and links within a post are denoted by the label urllink." 
This corpus is provided by http://u.cs.biu.ac.il/~schlerj/schler_springsymp06.pdf.

# Software
I am going to use python Keras (https://keras.io/) for development of this deep learning model. The main core of this model, since it is a convolutional nueral network (CNN), is the Conv1D layer from keras.layers. Other tentative libraries that i could use include Spacy, Pandas, nltk (for preprocessing), numpy e.t.c. They are to be used for calculating features sets called Basic-9 and Writeprints-static. I have also used scikitlearn for implementing machine learning based models.

# Literature Review
For literature review, i am going to take a look at the following research papers. They contain both the Deep learning model and machine learning model based approaches for solving the problem of authorship attribution. The two machine learning based model used in this project are Nueral networks with Basic-9 and Support vector machines with writeprints-static feature set. The deep learning model we use is a multi-channel CNN consisting of a static word embedding channel (word vectors trained by Word2Vec) and a non-static word embedding channel (word vectors trained initially by Word2Vec then updated during training).

https://www.cs.drexel.edu/~sa499/papers/anonymouth.pdf

https://www.aclweb.org/anthology/D14-1181

https://arxiv.org/abs/1609.06686

# Progress Milestones

# Design of Pipeline
I use Convolutional Neural Network (CNN) classifier with word embeddings for authorship attribution. More specifically, each word is mapped to a continuous-valued word vector using Word2Vec. Each input document is represented as a concatenation of word embeddings where each word embedding corresponds to a word in original document. The CNN model is trained using these document representations as input for authorship attribution. Then I train the multi-channel CNN consisting of a static word embedding channel (word vectors trained by Word2Vec) and a non-static word embedding channel (word vectors trained initially by Word2Vec then updated during training).
This feature set includes lexical and syntactic features.
For writeprints-static we have Lexical features which include character-level and word-level features such as total words, average word lenght, number of short words, total characters, percentage of digits, percentage of uppercase characters, special character occurances, letter frequency, digit frequency, character bigrams frequency, character trigrams frequency and some vocabulary richness features. 
Syntactic features include counts of function words (e.g., for, of), POS tags (e.g., Verb, Noun) and various punctuation (e.g., !;:). 
As suggested by literature review, these features are used with Support Vector Machine (SVM) classifier.
Basic - 9 Feature set used in this setting includes the following 9 features covering character-level, word-level and sentence-level features alongwith some readability metrics: character count (excluding whitespaces), number of unique words, lexical density (percentage of lexical words), average syllables per word, sentence count, average sentence length, Flesch-Kincaid readability metric, and Gunning-Fog readability metric. As suggested by literature review, we use the basic-9 feature set with a Feed Forward Neural Network (FFNN) whose number of layers were varied as a function of (number of features + number of target authors)/2.
# Implementation 
I have used the keras library for implementing the CNN based model and the machine learning based models are implemented using scikitlearn. The code is attached above.

# Results 
Following table summarizes the results for 5 authors and 10 authors setting in blogs dataset. It shows that the our CNN method outperforms the traditional Machine Learning methods. A 5 Author setting is applying authorship attribution classifier on 5 authors from blogs dataset which have 100 articles each and are selected according to the average number of characters per artcile. These 5 are the ones with highest averages. Same goes for the 10 author setting as well. The following table reports the classification accuracies.

|Setting   |Wrtiperints Static + SVM   |Basic-9 + FFNN   | CNN Method   |
|---|---|---|---|
|Blogs 5-Authors   |87%   |67%   |95%   |
|Blogs 10-Authors   |78%   |35%   |85%   |

As it can be seen, the deep learning model outperforms the machine learning based models. The reason in my point of view is the amount of data we have per author. Since we are using a 80-20 split, we have 80 articles per author to train and that gives the edge to CNN.
# Complete Report with Results (December 1st)
Applying this method on 10 Authors setting.

The code used in this method is an implementation of CNN-Word-Word model from https://arxiv.org/abs/1609.06686
