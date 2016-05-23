#!/usr/bin/env python

# hierarchical DL
# http://keras.io/models/about-keras-models/
# http://keras.io/getting-started/sequential-model-guide/
# model.load_weights('weight_file.h5')

import numpy as np
import pandas as pd

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.datasets import imdb
from keras.callbacks import Callback, EarlyStopping
from sklearn.cross_validation import train_test_split

from datetime import datetime

import variables_v2



    
def run(category):
    print 'Train posterior for the category: %s' %category
    
    nb_epochs = variables_v2.NB_EPOCHS
    batch_size = variables_v2.BATCH_SIZE
    lstm_output = variables_v2.LSTM_OUTPUT_SIZE
    lstm_dropout_w = variables_v2.LSTM_DROPOUT_W
    lstm_dropout_u = variables_v2.LSTM_DROPOUT_U
    dropout = variables_v2.DROPOUT
    patience = variables_v2.PATIENCE

    starttime = datetime.now()

    filepath = variables_v2.DATAPATH


    print 'Loading data...'
    reviews_positive = np.load(filepath + category + '/hdl_reviews_positive_embeddedall.npy')
    reviews_negative = np.load(filepath + category + '/hdl_reviews_negative_embeddedall.npy')
    ratings_positive = [1] * len(reviews_positive)
    ratings_negative = [0] * len(reviews_negative)

    reviews = np.append(reviews_positive, reviews_negative, axis=0)
    ratings = np.append(ratings_positive, ratings_negative, axis=0)
    ratings = ratings.reshape(len(reviews),1)

    del reviews_positive, reviews_negative, ratings_positive, ratings_negative

    seed = 1234
    np.random.seed(seed)
    np.random.shuffle(reviews)
    np.random.seed(seed)
    np.random.shuffle(ratings)
    
    X_train, X_test, y_train, y_test = train_test_split(reviews, ratings, test_size=0.2, random_state=42)

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    maxlen = X_train.shape[1]
    embedding_dim = X_train.shape[2]

    print '%d train sequences' %X_train.shape[0]
    print '%d test sequences' %X_test.shape[0]
    print 'maxlen: %d' %maxlen
    print 'embedding_dim: %d' %embedding_dim
    print

    print 'Builing model...'
    model = Sequential()
    model.add(LSTM(lstm_output, input_shape=(maxlen, embedding_dim,), dropout_W=lstm_dropout_w, dropout_U=lstm_dropout_u))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation('sigmoid')) # relu is worse, softmax does not work at all
    
    model.summary()

    model.load_weights(filepath + category + '/hdl/prior_keras_model_weights.h5')
    
    
    print 'Compiling model...'
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

        
    print 'Evaluating model...'
    score_train, acc_train = model.evaluate(X_train, y_train, batch_size=batch_size)
    print 'Train score: %f' %score_train
    print 'Train accuracy: %f' %acc_train
    score_test, acc_test = model.evaluate(X_test, y_test, batch_size=batch_size)
    print 'Test score: %f' %score_test
    print 'Test accuracy: %f' %acc_test
    
    
    endtime = datetime.now()
    print 'Runtime: %s' %(endtime - starttime)
    
    return score_train, acc_train, score_test, acc_test

if __name__ == '__main__':
    category = 'dvd'
    run(category)
    
    