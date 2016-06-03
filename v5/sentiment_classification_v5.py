#!/usr/bin/env python

'''sentiment_classification_v5.py

file description here
'''

__author__ = "Lorenzo von Ritter"
__date__ = "2016-06-03"

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback, EarlyStopping
from keras.optimizers import RMSprop
import cPickle as pkl
from datetime import datetime
import os
import variables_v5 as variables


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


def train_prior():
    print '\nTrain prior...'

    nb_epochs = variables.NB_EPOCHS_PRIOR
    batch_size = variables.BATCH_SIZE
    lstm_output = variables.LSTM_OUTPUT_SIZE
    lstm_dropout_w = variables.LSTM_DROPOUT_W
    lstm_dropout_u = variables.LSTM_DROPOUT_U
    dropout = variables.DROPOUT
    patience = variables.PATIENCE

    starttime = datetime.now()

    filepath = variables.DATAPATH

    print 'Loading data...'
    flag = 0
    for category in variables.CATEGORIES:
        print '\tcategory: %s' % category

        # read train and test reviews (words will be represented as integers)
        current_reviews_train = np.load(filepath + category + '/reviews_shuffled_embeddedIDa.npy')
        current_reviews_test = np.load(filepath + category + '/reviews_shuffled_embeddedIDb.npy')

        # read train and test ratings (1, 2, 4 or 5 stars)
        raw_ratings_train = np.genfromtxt(filepath + category + '/ratings_shuffleda')
        raw_ratings_test = np.genfromtxt(filepath + category + '/ratings_shuffledb')

        # transfrorm ratings into binary
        current_ratings_train = np.zeros(raw_ratings_train.shape)
        current_ratings_train[raw_ratings_train > 3] = 1
        current_ratings_train[raw_ratings_train < 3] = 0
        current_ratings_test = np.zeros(raw_ratings_test.shape)
        current_ratings_test[raw_ratings_test > 3] = 1
        current_ratings_test[raw_ratings_test < 3] = 0

        # make sure ratings are of shape (nb_samples, 1)
        current_ratings_train = current_ratings_train.reshape(len(current_reviews_train), 1)
        current_ratings_test = current_ratings_test.reshape(len(current_reviews_test), 1)

        # add reviews and ratings to global file
        if flag == 0:
            reviews_train = current_reviews_train
            reviews_test = current_reviews_test
            ratings_train = current_ratings_train
            ratings_test = current_ratings_test
            flag = 1

        else:
            reviews_train = np.append(reviews_train, current_reviews_train, axis=0)
            reviews_test = np.append(reviews_test, current_reviews_test, axis=0)
            ratings_train = np.append(ratings_train, current_ratings_train, axis=0)
            ratings_test = np.append(ratings_test, current_ratings_test, axis=0)

    seed = 1234
    np.random.seed(seed)
    np.random.shuffle(reviews_train)
    np.random.seed(seed)
    np.random.shuffle(reviews_test)
    np.random.seed(seed)
    np.random.shuffle(ratings_train)
    np.random.seed(seed)
    np.random.shuffle(ratings_test)

    X_train = reviews_train
    X_test = reviews_test
    y_train = ratings_train
    y_test = ratings_test

    del reviews_train, reviews_test
    del ratings_train, ratings_test

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    print '%d train sequences' % X_train.shape[0]
    print '%d test sequences' % X_test.shape[0]

    maxlen = X_train.shape[1]
    assert maxlen == variables.MAX_LENGTH
    embedding_dim = variables.EMBEDDING_DIM

    print 'maxlen: %d' %maxlen
    print 'embedding_dim: %d' %embedding_dim

    print('Setting up Arrays for Keras Embedding Layer...')
    with open(filepath + 'all/reviews.vocab.pkl') as indexdictfile:
        index_dict = pkl.load(indexdictfile)
    with open(filepath + 'all/reviews.vectors.pkl') as glovedictfile:
        glove_dict = pkl.load(glovedictfile)

    # adding 1 to account for 0th index that is not included in index_dict
    nb_words = len(index_dict) + 1

    # create a matrix of shape (nb_words, embedding_dim) that lists the embedding for each word
    embedding_weights = np.zeros((nb_words, embedding_dim))
    for word, index in index_dict.items():
        embedding_weights[index, :] = glove_dict[word]

    print 'Builing model...'
    model = Sequential()
    model.add(Embedding(input_dim=nb_words,
                        input_length=maxlen,
                        output_dim=embedding_dim,
                        mask_zero=True,                 # mask 0 (unknwon word)
                        weights=[embedding_weights],
                        trainable=False))               # prior Embedding is static
                        # dropout=dropout_embedding))  # added dropout
    model.add(
        LSTM(lstm_output, input_shape=(maxlen, embedding_dim,), dropout_W=lstm_dropout_w, dropout_U=lstm_dropout_u))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))  # relu is worse, softmax does not work at all

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
    print '\tTrain score: %f' % score
    print '\tTrain accuracy: %f' % acc
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    print '\tTest score: %f' % score
    print '\tTest accuracy: %f' % acc

    savepath = filepath + 'all/v5/keras_' + setting_prior
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    print 'Saving history...'
    historyfile = pd.DataFrame(data=np.transpose([history.loss, history.acc, history.val_loss, history.val_acc]),
                               columns=['loss', 'acc', 'val_loss', 'val_acc'])
    historyfile.to_csv(savepath + '/keras_history.csv', index=False)

    print 'Saving model...'
    json_string = model.to_json()
    open(savepath + '/keras_model_architecture.json', 'w').write(json_string)
    model.save_weights(savepath + '/keras_model_weights.h5', overwrite=True)

    del model  # delete model to be sure that to omit overfitting

    endtime = datetime.now()
    print 'Runtime: %s' % (endtime - starttime)


def train_posterior(category):
    print '\nTrain posterior for the category: %s' % category

    nb_epochs = variables.NB_EPOCHS_POSTERIOR  # 60 # variables_v2.NB_EPOCHS
    batch_size = variables.BATCH_SIZE
    lstm_output = variables.LSTM_OUTPUT_SIZE
    lstm_dropout_w = variables.LSTM_DROPOUT_W
    lstm_dropout_u = variables.LSTM_DROPOUT_U
    dropout = variables.DROPOUT
    patience = variables.PATIENCE

    starttime = datetime.now()

    filepath = variables.DATAPATH

    print 'Loading data...'
    # read train and test reviews (words will be represented as integers)
    reviews_train = np.load(filepath + category + '/reviews_shuffled_embeddedIDa.npy')
    reviews_test = np.load(filepath + category + '/reviews_shuffled_embeddedIDb.npy')

    # read train and test ratings (1, 2, 4 or 5 stars)
    raw_ratings_train = np.genfromtxt(filepath + category + '/ratings_shuffleda')
    raw_ratings_test = np.genfromtxt(filepath + category + '/ratings_shuffledb')

    # transfrorm ratings into binary
    ratings_train = np.zeros(raw_ratings_train.shape)
    ratings_train[raw_ratings_train > 3] = 1
    ratings_train[raw_ratings_train < 3] = 0
    ratings_test = np.zeros(raw_ratings_test.shape)
    ratings_test[raw_ratings_test > 3] = 1
    ratings_test[raw_ratings_test < 3] = 0

    # make sure ratings are of shape (nb_samples, 1)
    ratings_train = ratings_train.reshape(len(reviews_train), 1)
    ratings_test = ratings_test.reshape(len(reviews_test), 1)

    X_train = reviews_train
    X_test = reviews_test
    y_train = ratings_train
    y_test = ratings_test

    del reviews_train, reviews_test
    del ratings_train, ratings_test

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    print '%d train sequences' % X_train.shape[0]
    print '%d test sequences' % X_test.shape[0]

    maxlen = X_train.shape[1]
    assert maxlen == variables.MAX_LENGTH
    embedding_dim = variables.EMBEDDING_DIM

    print 'maxlen: %d' %maxlen
    print 'embedding_dim: %d' %embedding_dim

    print('Setting up Arrays for Keras Embedding Layer...')
    with open(filepath + 'all/reviews.vocab.pkl') as indexdictfile:
        index_dict = pkl.load(indexdictfile)
    with open(filepath + 'all/reviews.vectors.pkl') as glovedictfile:
        glove_dict = pkl.load(glovedictfile)

    # adding 1 to account for 0th index that is not included in index_dict
    nb_words = len(index_dict) + 1

    # create a matrix of shape (nb_words, embedding_dim) that lists the embedding for each word
    embedding_weights = np.zeros((nb_words, embedding_dim))
    for word, index in index_dict.items():
        embedding_weights[index, :] = glove_dict[word]

    print 'Builing model...'
    model = Sequential()
    model.add(Embedding(input_dim=nb_words,
                        input_length=maxlen,
                        output_dim=embedding_dim,
                        mask_zero=True,                 # mask 0 (unknwon word)
                        weights=[embedding_weights],
                        dropout=variables.DROPOUT_EMBEDDING,  # added dropout
                        trainable=True))               # posterior Embedding is trainable
    model.add(
        LSTM(lstm_output, input_shape=(maxlen, embedding_dim,), dropout_W=lstm_dropout_w, dropout_U=lstm_dropout_u))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))  # relu is worse, softmax does not work at all

    model.summary()

    model.load_weights(filepath + 'all/v5/keras_' + setting_prior + '/keras_model_weights.h5')

    print 'Compiling model...'
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop', # RMSprop(lr=1e-6), #TODO: decrease learning rate (default is 1e-3)
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
    print '\tTrain score: %f' % score
    print '\tTrain accuracy: %f' % acc
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    print '\tTest score: %f' % score
    print '\tTest accuracy: %f' % acc

    savepath = filepath + category + '/v5/keras_' + setting_posterior
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    print 'Saving history...'
    historyfile = pd.DataFrame(data=np.transpose([history.loss, history.acc, history.val_loss, history.val_acc]),
                               columns=['loss', 'acc', 'val_loss', 'val_acc'])
    historyfile.to_csv(savepath + '/keras_history.csv', index=False)

    print 'Saving model...'
    json_string = model.to_json()
    open(savepath + '/keras_model_architecture.json', 'w').write(json_string)
    model.save_weights(savepath + '/keras_model_weights.h5', overwrite=True)

    del model  # delete model to be sure that to omit overfitting

    endtime = datetime.now()
    print 'Runtime: %s' % (endtime - starttime)


if __name__ == '__main__':
    global_starttime = datetime.now()

    for learning_rate in [1e-4, 1e-5, 1e-6]:
        setting_prior = ''
        setting_posterior = 'lr%s' %learning_rate

        #train_prior()

        for category in variables.CATEGORIES:
            train_posterior(category)

    global_endtime = datetime.now()
    print 'Global runtime %s' % (global_endtime - global_starttime)