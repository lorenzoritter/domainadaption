#!/usr/bin/env python


import numpy as np
import cPickle as pkl
import variables_v2


def read_input(infilename,outfilename, w2v_dict, max_length, embedding_dim):
    print '\tEmbedding %s...' %infilename
    embedded_reviews = []
    
    with open(infilename, 'r') as infile:
        for onereview in infile:
            wordindex = 0
            embedded_onereview = np.zeros((max_length, embedding_dim))
            for word in onereview.split():
                if wordindex >= max_length: # break if the max length is reached
                    break
                if word == '<eos>':  # replace <eos> tag with ones
                    embedded_onereview[wordindex] = np.ones(embedding_dim)
                else:
                    try:
                        embedded_onereview[wordindex] = w2v_dict[word]                    
                    except KeyError:   # if a key error happens (word not in vocabulary), pad with zeros
                        embedded_onereview[wordindex] = np.zeros(embedding_dim)
                wordindex += 1
            embedded_reviews.append(embedded_onereview)
    np.save(outfilename, embedded_reviews)
    

filepath = variables_v2.DATAPATH
max_length = variables_v2.MAX_LENGTH
embedding_dim = variables_v2.EMBEDDING_DIM
glove_dict_embeddedall = pkl.load(open(filepath + 'all/reviews.vectors.pkl'))

for sentiment in variables_v2.SENTIMENTS:
    print 'sentiment: ' + sentiment
    for category in variables_v2.CATEGORIES:
        print '\tcategory: ' + category
        infilename = filepath + category + '/reviews_' + sentiment + '.txt'
        outfilename = filepath + category + '/hdl_reviews_' + sentiment + '_embeddedall.npy'
        read_input(infilename, outfilename, glove_dict_embeddedall, max_length=max_length, embedding_dim=embedding_dim)    

        infilename = filepath + category + '/allno' + category + '_reviews_' + sentiment + '.txt'
        outfilename = filepath + category + '/hdl_allno' + category + '_reviews_' + sentiment + '_embeddedall.npy'
        read_input(infilename, outfilename, glove_dict_embeddedall, max_length=max_length, embedding_dim=embedding_dim)
        
        
            
            
            
            
