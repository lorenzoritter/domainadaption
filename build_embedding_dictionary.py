#!/usr/bin/env python

'''build_embedding_dictionary.py

file description here
'''

__author__ = "Lorenzo von Ritter"
__date__ = "2016 - 05 - 09"

import cPickle as pkl
categories = ['books', 'dvd', 'electronics', 'kitchen_&_housewares', 'all']

filepath = '/home/lorenzo/PycharmProjects/domainadaption/sorted_data_acl/'

for category in categories:
    print 'category: ' + category

    glove_dict = {}     # the dictionary is going to be of shape word -> embedding vector

    # read vectors into dictionary
    with open(filepath + category + '/reviews.vectors.txt') as w2v:
        for line in w2v:
            words = line.split()
            glove_dict[words[0]] = map(float, words[1:])  # add dictionary entry for the new word

    # save dictionary
    with open(filepath + category + '/reviews.vectors.pkl', 'wb') as dictfile:
        pkl.dump(glove_dict, dictfile)