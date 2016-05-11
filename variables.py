#!/usr/bin/env python

'''variables.py

file description here
'''

__author__ = "Lorenzo von Ritter"
__date__ = "2016-05-11"

import os

EMBEDDING_DIM = 250     # dimension of GloVe word vectors
MAX_LENGTH = 200        # maximum length of reviews (longer reviews are cut, shorter reviews are padded)
NB_EPOCHS = 1000        # number of training epochs of the neural network
BATCH_SIZE = 32         # batch size during training of the neural network
LSTM_OUTPUT_SIZE = 50   # number of output neurons of the LSTM
LSTM_DROPOUT_W = 0.5    # dropout fraction at the input nodes of the LSTM
LSTM_DROPOUT_U = 0.1    # dropout fraction at the recurrent connections of the LSTM
DROPOUT = 0.5           # dropout fraction at the dropout layer

os.environ['EMBEDDING_DIM'] = str(EMBEDDING_DIM)
os.environ['MAX_LENGTH'] = str(MAX_LENGTH)