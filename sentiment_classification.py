#!/usr/bin/env python

'''sentiment_classification.py

Train a LSTM on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF+LogReg.

Notes:

    - RNNs are tricky. Choice of batch size is important,
    choice of loss and optimizer is critical, etc.
    Some configurations won't converge.

    - LSTM loss decrease patterns during training can be quite different
    from what you see with CNNs/MLPs/etc.

GPU command:
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm.py
'''

__author__ = "Lorenzo von Ritter"
__date__ = "2016-05-10"


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

#categories = ['books', 'dvd', 'electronics', 'kitchen_&_housewares', 'all']
categories = ['books']
#sentiments = ['positive', 'negative']

filepath = '/home/lorenzo/PycharmProjects/domainadaption/sorted_data_acl/'


for category in categories:
    print 'category: %s' %category

    print 'Loading data...'
    reviews_positive = np.load(filepath + category + '/reviews_positive.npy')
    reviews_negative = np.load(filepath + category + '/reviews_negative.npy')
    #ratings_positive = np.load(filepath + category + '/ratings_positive.npy')
    #ratings_negative = np.load(filepath + category + '/ratings_negative.npy')
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

    print '%d train sequences' %X_train.shape[0]
    print '%d test sequences' %X_test.shape[0]

    maxlen = X_train.shape[1]
    embedding_dim = X_train.shape[2]

    print 'maxlen: %d' %maxlen
    print 'embedding_dim: %d' %embedding_dim

    '''
    print 'Pad sequences (samples x time)'
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print 'X_train shape: %s' %X_train.shape
    print 'X_test shape: %s' %X_test.shape
    '''

    print 'Builing model...'
    model = Sequential()
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
