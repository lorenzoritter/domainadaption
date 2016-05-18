#!/usr/bin/env python

'''build_integer_dictionary.py

file description here
'''

__author__ = "Lorenzo von Ritter"
__date__ = "2016-05-18"

import cPickle as pkl
import variables

categories = ['books', 'dvd', 'electronics', 'kitchen_&_housewares', 'all']
filepath = variables.DATAPATH

for category in categories:
    print 'category: ' + category

    integer_dict = {}     # the dictionary is going to be of shape word -> embedding vector

    # read vectors into dictionary
    with open(filepath + category + '/reviews.vocab') as w2v:

        word_id = 1

        for line in w2v:
            words = line.split()
            integer_dict[words[0]] = word_id  # add dictionary entry for the new word
            word_id += 1

    # save dictionary
    with open(filepath + category + '/reviews.vocab.pkl', 'wb') as dictfile:
        pkl.dump(integer_dict, dictfile)