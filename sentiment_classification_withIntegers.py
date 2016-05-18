#!/usr/bin/env python

'''sentiment_classification_withIntegers.py

file description here
'''

__author__ = "Lorenzo von Ritter"
__date__ = "2016-05-18"


import numpy as np
import pandas as pd
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.datasets import imdb
from keras.callbacks import Callback, EarlyStopping
from sklearn.cross_validation import train_test_split
import cPickle as pkl
from datetime import datetime

import variables


# class to store the loss at each end of each epoch
class TrainHistory(Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))


#max_features = 20000
#maxlen = 200  # cut texts after this number of words (among top max_features most common words)
nb_epochs = variables.NB_EPOCHS
batch_size = variables.BATCH_SIZE
lstm_output = variables.LSTM_OUTPUT_SIZE
lstm_dropout_w = variables.LSTM_DROPOUT_W
lstm_dropout_u = variables.LSTM_DROPOUT_U
dropout = variables.DROPOUT
patience = variables.PATIENCE

starttime = datetime.now()

categories = ['all', 'books', 'dvd', 'electronics', 'kitchen_&_housewares']
#sentiments = ['positive', 'negative']

filepath = variables.DATAPATH


for category in categories:
    print 'category: %s' %category

    print 'Loading data...'
    reviews = np.load(filepath + category + '/reviews_all.npy')

    raw_ratings = np.genfromtxt(filepath + category + '/ratings_all.txt', dtype=np.int32)
    ratings = np.zeros(raw_ratings.shape)
    ratings[raw_ratings > 3] = 1
    ratings[raw_ratings < 3] = 0

    seed = 1234
    np.random.seed(seed)
    np.random.shuffle(reviews)
    np.random.seed(seed)
    np.random.shuffle(ratings)

    X_train, X_test, y_train, y_test = train_test_split(reviews, ratings, test_size=0.2, random_state=42)

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    print '%d train sequences' %X_train.shape[0]
    print '%d test sequences' %X_test.shape[0]

    maxlen = X_train.shape[1]
    assert maxlen == variables.MAX_LENGTH
    embedding_dim = variables.EMBEDDING_DIM

    print 'maxlen: %d' %maxlen
    print 'embedding_dim: %d' %embedding_dim

    print('Setting up Arrays for Keras Embedding Layer...')
    with open(filepath + category + '/reviews.vocab.pkl') as indexdictfile:
        index_dict = pkl.load(indexdictfile)
    with open(filepath + category + '/reviews.vectors.pkl') as glovedictfile:
        glove_dict = pkl.load(glovedictfile)

    nb_words = len(index_dict) + 1  # adding 1 to account for 0th index

    embedding_weights = np.zeros((nb_words, embedding_dim))
    for word, index in index_dict.items():
        embedding_weights[index, :] = glove_dict[word]


    print 'Builing model...'
    model = Sequential()
    model.add(Embedding(input_dim=nb_words,
                        input_length=maxlen,  # Adding Input Length
                        output_dim=embedding_dim,
                        mask_zero=True,
                        weights=[embedding_weights]))
                        # dropout=0.5))  # added dropout
    model.add(LSTM(lstm_output, input_shape=(maxlen, embedding_dim,), dropout_W=lstm_dropout_w, dropout_U=lstm_dropout_u))
    #model.add(LSTM(lstm_output, input_shape=(maxlen, embedding_dim,)))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation('sigmoid')) # relu is worse, softmax does not work at all

    model.summary()

    # try using different optimizers and different optimizer configs
    print 'Compiling model...'
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # initialize callbacks
    history = TrainHistory()
    early_stopping = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=patience)

    print 'Training model...'
    try:
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epochs,
                  validation_data=(X_test, y_test), callbacks=[history])
    except KeyboardInterrupt:
        print '\nTraining interrupted by user. Continuing with model evaluation...'

    print 'Evaluating model...'
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    print 'Test score: %f' %score
    print 'Test accuracy: %f' %acc

    print 'Saving history...'
    historyfile = pd.DataFrame(data=np.transpose([history.loss, history.acc, history.val_loss, history.val_acc]),
                               columns=['loss', 'acc', 'val_loss', 'val_acc'])
    historyfile.to_csv(filepath + category + '/keras_history.csv', index=False)

    print 'Saving model...'
    json_string = model.to_json()
    open(filepath + category + '/keras_model_architecture.json', 'w').write(json_string)
    model.save_weights(filepath + category + '/keras_model_weights.h5', overwrite=True)

    endtime = datetime.now()
    print 'Runtime: %s' %(endtime - starttime)
