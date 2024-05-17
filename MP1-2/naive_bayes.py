# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter
import nltk
from nltk.corpus import stopwords


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naiveBayes(dev_set, train_set, train_labels, laplace=0.01, pos_prior=0.5, silently=False):
    print_values(laplace,pos_prior)

    yhats = []
    tot = {}
    pos = {}
    pos_num = 0
    neg = {}
    neg_num = 0

    nltk.download('stopwords')
    stop_words = stopwords.words('english')

    ##set train_set and dev_set to lower case
    # for i in range(len(train_set)):
    #     for k in range(len(train_set[i])):
    #         train_set[i][k] = train_set[i][k].lower()
    # for i in range(len(dev_set)):
    #     for k in range(len(dev_set[i])):
    #         dev_set[i][k] = dev_set[i][k].lower()

    #delete stop words from train_set
    a = []
    for i in train_set:
        b = []
        for k in i:
            if k not in stop_words and len(k):
                b.append(k)
        a.append(b)
    train_set = a

    #delete stop words from dev_set
    a = []
    for i in dev_set:
        b = []
        for k in i:
            if k not in stop_words and len(k):
                b.append(k)
        a.append(b)
    dev_set = a

    #training model
    for i in range(len(train_labels)):
        x = Counter(train_set[i])
        if train_labels[i] == 1:
            for k in x.keys():
                tot[k] = 1
                pos_num += x[k]
                if k in pos.keys():
                    pos[k] += x[k]
                else:
                    pos[k] = x[k]
        else:
            for k in x.keys():
                tot[k] = 1
                neg_num += x[k]
                if k in neg.keys():
                    neg[k] += x[k]
                else:
                    neg[k] = x[k]

    #develop model
    for doc in tqdm(dev_set, disable=silently):
        pos_prob = math.log(pos_prior)
        neg_prob = math.log(1-pos_prior)
        tot_V = len(tot.keys())

        for i in range(len(doc)):
            if (doc[i] in pos.keys()):
                pos_prob += math.log((pos[doc[i]]+laplace)/(pos_num+laplace*(tot_V+1)))
            else:
                pos_prob += math.log(laplace/(pos_num+laplace*(tot_V+1)))
            if (doc[i] in neg.keys()):
                neg_prob += math.log((neg[doc[i]]+laplace)/(neg_num+laplace*(tot_V+1)))
            else:
                neg_prob += math.log(laplace/(neg_num+laplace*(tot_V+1)))
        if pos_prob > neg_prob:
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats
