#!/usr/bin/env python

'''embed_reviews.py

file description here
'''

__author__ = "Lorenzo von Ritter"
__date__ = "2016-05-09"

import numpy as np
import cPickle as pkl
import variables

categories = ['books', 'dvd', 'electronics', 'kitchen_&_housewares', 'all']
sentiments = ['positive', 'negative']

filepath = '/home/lorenzo/PycharmProjects/domainadaption/sorted_data_acl/'

max_length = variables.MAX_LENGTH
embedding_dim = variables.EMBEDDING_DIM

def read_input(filename, w2v_dict, max_length, embedding_dim):
    print '\tEmbedding %s...' %filename
    embedded_file = []
    #for review in open(os.path.join(dirname, fname)): # could combine the following two lines
    with open(filename, 'r') as infile:
        #lineindex = 0
        for review in infile:
            #infile_data = infile.read()
            wordindex = 0
            embedded_line = np.zeros((max_length, embedding_dim))
            for word in review.split():
                # if the maximum length of words is reached, break
                if wordindex >= max_length:
                    break
                try:
                    embedded_line[wordindex] = w2v_dict[word]
                    wordindex += 1
                # if a key error happens, pass to next word (maybe consider a special character for unknown words)
                except:
                    pass
            #embedded_file[lineindex] = np.array([word2vec_model[word] for word in review.split()])
            #lineindex += 1
            embedded_file.append(embedded_line)
            #print embedded_file
    return np.array(embedded_file)


for category in categories:
    print 'category: ' + category

    with open(filepath + category + '/reviews.vectors.pkl') as dictfile:
        glove_dict = pkl.load(dictfile)

    for sentiment in sentiments:
        print '\t' + sentiment
        infilename = filepath + category + '/reviews_' + sentiment + '.txt'
        embedded_reviews = read_input(infilename, glove_dict, max_length=max_length, embedding_dim=embedding_dim)

        outfilename = filepath + category + '/reviews_' + sentiment + '.npy'
        np.save(outfilename, embedded_reviews)
