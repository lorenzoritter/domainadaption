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

        
def train_prior(category):
    print 'Train prior for the category: %s' %category
    
    nb_epochs = variables_v2.NB_EPOCHS_PRIOR
    batch_size = variables_v2.BATCH_SIZE
    lstm_output = variables_v2.LSTM_OUTPUT_SIZE
    lstm_dropout_w = variables_v2.LSTM_DROPOUT_W
    lstm_dropout_u = variables_v2.LSTM_DROPOUT_U
    dropout = variables_v2.DROPOUT
    patience = variables_v2.PATIENCE

    starttime = datetime.now()

    filepath = variables_v2.DATAPATH


    print 'Loading data...'
    reviews_positive = np.load(filepath + category + '/hdl_allno' + category + '_reviews_positive_embeddedall.npy')
    reviews_negative = np.load(filepath + category + '/hdl_allno' + category + '_reviews_negative_embeddedall.npy')
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
    
    # test data will be used for validation
    X_train, X_test, y_train, y_test = train_test_split(reviews, ratings, test_size=0.1, random_state=42)

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
    
    print 'Compiling model...'
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    history = TrainHistory()  # initialize callbacks
    early_stopping = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=patience)

    
    print 'Training model...'
    try:
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epochs,
                  validation_data=(X_test, y_test), callbacks=[history])
    except KeyboardInterrupt:
        print '\nTraining interrupted by user. Continuing with model evaluation...'

        
    print 'Evaluating model...'
    score, acc = model.evaluate(X_train, y_train, batch_size=batch_size)
    print 'Train score: %f' %score
    print 'Train accuracy: %f' %acc    
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    print 'Test score: %f' %score
    print 'Test accuracy: %f' %acc
    
    
    print 'Saving history...'
    historyfile = pd.DataFrame(data=np.transpose([history.loss, history.acc, history.val_loss, history.val_acc]),
                               columns=['loss', 'acc', 'val_loss', 'val_acc'])
    historyfile.to_csv(filepath + category + '/hdl/prior_keras_history.csv', index=False)

    print 'Saving model...'
    json_string = model.to_json()
    open(filepath + category + '/hdl/prior_keras_model_architecture.json', 'w').write(json_string)
    model.save_weights(filepath + category + '/hdl/prior_keras_model_weights.h5', overwrite=True)

    endtime = datetime.now()
    print 'Runtime: %s' %(endtime - starttime)

    
def train_posterior(category):
    print 'Train posterior for the category: %s' %category
    
    nb_epochs = variables_v2.NB_EPOCHS_POSTERIOR #60 # variables_v2.NB_EPOCHS
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

    history = TrainHistory()  # initialize callbacks
    early_stopping = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=patience)

    
    print 'Training model...'
    try:
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epochs,
                  validation_data=(X_test, y_test), callbacks=[history])
    except KeyboardInterrupt:
        print '\nTraining interrupted by user. Continuing with model evaluation...'

        
    print 'Evaluating model...'
    score, acc = model.evaluate(X_train, y_train, batch_size=batch_size)
    print 'Train score: %f' %score
    print 'Train accuracy: %f' %acc    
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    print 'Test score: %f' %score
    print 'Test accuracy: %f' %acc
    
    
    
    print 'Saving history...'
    historyfile = pd.DataFrame(data=np.transpose([history.loss, history.acc, history.val_loss, history.val_acc]),
                               columns=['loss', 'acc', 'val_loss', 'val_acc'])
    historyfile.to_csv(filepath + category + '/hdl/posterior_keras_history.csv', index=False)

    print 'Saving model...'
    json_string = model.to_json()
    open(filepath + category + '/hdl/posterior_keras_model_architecture.json', 'w').write(json_string)
    model.save_weights(filepath + category + '/hdl/posterior_keras_model_weights.h5', overwrite=True)

    endtime = datetime.now()
    print 'Runtime: %s' %(endtime - starttime)

    
if __name__ == '__main__':
    for category in variables_v2.CATEGORIES:
        train_prior(category)
        train_posterior(category)
    
    