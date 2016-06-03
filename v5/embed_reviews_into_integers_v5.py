#!/usr/bin/env python

'''embed_reviews_into_integers_v5.py

Transform words of reviews into their corresponding IDs according to the global embedding.
'''

__author__ = "Lorenzo von Ritter"
__date__ = "2016-06-03"


import numpy as np
import cPickle as pkl
import variables


filepath = variables.DATAPATH
categories = ['books', 'dvd', 'electronics', 'kitchen']
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
    return np.array(embedded_file) # .astype(np.uint32) # maximum allowed vocabulary 4e9

def embed_into_ids():
    # read global embedding
    with open(filepath + 'all/reviews.vocab.pkl') as dictfile:
        integer_dict = pkl.load(dictfile)

    for category in categories:
        print 'category: ' + category

        # use the shuffled reviews
        for part in ['a', 'b']:
            infilename = filepath + category + '/reviews_shuffled' + part
            outfilename = filepath + category + '/reviews_shuffled_embeddedID' + part + '.npy'

            embedded_reviews = read_input(infilename, integer_dict, max_length=max_length)

            np.save(outfilename, embedded_reviews)

if __name__ == '__main__':
    embed_into_ids()