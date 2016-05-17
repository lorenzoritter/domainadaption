#!/usr/bin/env python

'''embed_reviews_hdf5.py

Convert words in all reviews into embeddings and store them in HDF5 format.
'''

__author__ = "Lorenzo von Ritter"
__date__ = "2016-05-17"

import numpy as np
import cPickle as pkl
import h5py
import os
import variables

categories = ['books', 'dvd', 'electronics', 'kitchen_&_housewares', 'all']
sentiments = ['all'] #['positive', 'negative']

filepath = variables.DATAPATH

max_length = variables.MAX_LENGTH
embedding_dim = variables.EMBEDDING_DIM

def read_input(infilename, outfilename, w2v_dict, max_length, embedding_dim):
    print '\tEmbedding %s...' %infilename

    if os.path.exists(outfilename):
        os.remove(outfilename)

    with open(infilename, 'r') as infile, h5py.File(outfilename, 'a') as outfile:
        dset = outfile.create_dataset('embedded_reviews', shape=(0,variables.MAX_LENGTH,variables.EMBEDDING_DIM),
                                      maxshape=(None,variables.MAX_LENGTH,variables.EMBEDDING_DIM))
        #chunksize = 1000
        #cunk = []

        reviewindex = 0

        for review in infile:
            #infile_data = infile.read()
            wordindex = 0
            embedded_line = np.zeros((max_length, embedding_dim))
            for word in review.split():
                # if the maximum length of words is reached, break
                if wordindex >= max_length:
                    break
                # replace <eos> tag with ones
                if word == '<eos>':
                    embedded_line[wordindex] = np.ones(embedding_dim)
                else:
                    try:
                        embedded_line[wordindex] = w2v_dict[word]
                    # if a key error happens (word not in vocabulary), pad with zeros
                    except KeyError:
                        embedded_line[wordindex] = np.zeros(embedding_dim)
                wordindex += 1

            # TODO: Do this in chunks instead of one by one
            dset.resize((reviewindex+1,variables.MAX_LENGTH,variables.EMBEDDING_DIM))
            dset[reviewindex,:,:] = embedded_line

            reviewindex += 1


for category in categories:
    print 'category: ' + category

    with open(filepath + category + '/reviews.vectors.pkl') as dictfile:
        glove_dict = pkl.load(dictfile)

    for sentiment in sentiments:
        print '\t' + sentiment
        infilename = filepath + category + '/reviews_' + sentiment + '.txt'
        outfilename = filepath + category + '/reviews_' + sentiment + '.h5'

        read_input(infilename, outfilename, glove_dict, max_length=max_length, embedding_dim=embedding_dim)