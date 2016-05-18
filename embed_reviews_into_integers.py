#!/usr/bin/env python

'''embed_reviews_into_integers.py

file description here
'''

__author__ = "Lorenzo von Ritter"
__date__ = "2016-05-18"


import numpy as np
import cPickle as pkl
import variables

categories = ['books', 'dvd', 'electronics', 'kitchen_&_housewares', 'all'] #['books', 'dvd', 'electronics', 'kitchen_&_housewares', 'all']
sentiments = ['all'] #['positive', 'negative']

filepath = variables.DATAPATH

max_length = variables.MAX_LENGTH

def read_input(infilename, integer_dict, max_length):
    print '\tEmbedding %s...' %infilename

    embedded_file = []

    with open(infilename, 'r') as infile:

        for review in infile:

            wordindex = 0
            embedded_line = np.zeros(max_length)

            for word in review.split():
                # if the maximum length of words is reached, break
                if wordindex >= max_length:
                    break
                try:
                    embedded_line[wordindex] = integer_dict[word]
                # if a key error happens (word not in vocabulary), pad with zeros
                except KeyError:
                    embedded_line[wordindex] = 0
                wordindex += 1

            embedded_file.append(embedded_line)
    return np.array(embedded_file)


for category in categories:
    print 'category: ' + category

    with open(filepath + category + '/reviews.vocab.pkl') as dictfile:
        integer_dict = pkl.load(dictfile)

    for sentiment in sentiments:
        print '\t' + sentiment
        infilename = filepath + category + '/reviews_' + sentiment + '.txt'
        outfilename = filepath + category + '/reviews_' + sentiment + '.npy'

        embedded_reviews = read_input(infilename, integer_dict, max_length=max_length)

        np.save(outfilename, embedded_reviews)
