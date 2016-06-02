#!/usr/bin/env python

'''variables_v4.py

file description here
'''

__author__ = "Lorenzo von Ritter"
__date__ = "2016-06-02"

DATAPATH = '/home/lorenzo/PycharmProjects/domainadaption/sorted_data_acl/'
CATEGORIES = ['books', 'dvd', 'electronics', 'kitchen'] #['dvd', 'electronics', 'kitchen_&_housewares'] # ['books', 'dvd', 'electronics', 'kitchen_&_housewares']
SENTIMENTS = ['all'] # ['positive', 'negative'] # ['positive', 'negative', 'unlabeled]

EMBEDDING_DIM = 250     # dimension of GloVe word vectors
MAX_LENGTH = 200        # maximum length of reviews (longer reviews are cut, shorter reviews are padded)
NB_EPOCHS = 500         # number of training epochs of the neural network
NB_EPOCHS_PRIOR=200
NB_EPOCHS_POSTERIOR=200
BATCH_SIZE = 32         # batch size during training of the neural network
DROPOUT_EMBEDDING = 0.5 # dropout at Embedding layer
LSTM_OUTPUT_SIZE = 50   # number of output neurons of the LSTM
LSTM_DROPOUT_W = 0.5    # dropout fraction at the input nodes of the LSTM
LSTM_DROPOUT_U = 0.3    # dropout fraction at the recurrent connections of the LSTM
DROPOUT = 0.8           # dropout fraction at the dropout layer
PATIENCE = 20           # patience for early stopping: number of worse epochs before stopping training
MAX_REVIEWS = 19800     # maximum amount of reviews to parse from the raw files
