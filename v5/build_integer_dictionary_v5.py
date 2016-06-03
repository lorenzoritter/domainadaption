#!/usr/bin/env python

'''build_integer_dictionary_v5.py

Creates a word->ID dictionary from the GloVe output. The list will be orderd from the most frequent to the least
frequent word. The ID 0 is not assigned, since it is reserved for unknown words.
'''

__author__ = "Lorenzo von Ritter"
__date__ = "2016-06-03"

import cPickle as pkl
import variables

categories = ['all']
filepath = variables.DATAPATH

for category in categories:
    print 'category: ' + category

    integer_dict = {}     # the dictionary is going to be of shape word -> word ID

    # read vocabulary into dictionary
    with open(filepath + category + '/reviews.vocab') as vocabulary:

        word_id = 1

        for line in vocabulary:
            words = line.split()
            integer_dict[words[0]] = word_id  # add dictionary entry for the new word
            word_id += 1

    # save dictionary
    with open(filepath + category + '/reviews.vocab.pkl', 'wb') as dictfile:
        pkl.dump(integer_dict, dictfile)