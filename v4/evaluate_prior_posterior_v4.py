#!/usr/bin/env python

'''evaluate_prior_posterior_v4.py

file description here
'''

__author__ = "Lorenzo von Ritter"
__date__ = "2016-06-02"

import numpy as np
from keras.models import model_from_json
import variables_v4 as variables


def evaluate_prior(setting):
    print '\nEvaluating prior...'
    datapath = variables.DATAPATH
    categories = variables.CATEGORIES
    batch_size = variables.BATCH_SIZE

    modelpath = datapath + 'all/v4/keras_' + setting + '/'

    model = model_from_json(open(modelpath + 'keras_model_architecture.json').read())
    model.load_weights(modelpath + 'keras_model_weights.h5')

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    for category in categories:
        print '\ncategory: ' + category
        current_reviews_train = np.load(datapath + category + '/hdl_reviews_shuffled_embeddedalla.npy')
        current_reviews_test = np.load(datapath + category + '/hdl_reviews_shuffled_embeddedallb.npy')

        raw_ratings_train = np.genfromtxt(datapath + category + '/ratings_shuffleda')
        raw_ratings_test = np.genfromtxt(datapath + category + '/ratings_shuffledb')

        current_ratings_train = np.zeros(raw_ratings_train.shape)
        current_ratings_train[raw_ratings_train > 3] = 1
        current_ratings_train[raw_ratings_train < 3] = 0
        current_ratings_test = np.zeros(raw_ratings_test.shape)
        current_ratings_test[raw_ratings_test > 3] = 1
        current_ratings_test[raw_ratings_test < 3] = 0

        del raw_ratings_train, raw_ratings_test

        X_train = current_reviews_train
        X_test = current_reviews_test
        y_train = current_ratings_train
        y_test = current_ratings_test

        del current_reviews_train, current_reviews_test
        del current_ratings_train, current_ratings_test

        print 'Evaluate training data...'
        score, acc = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
        print "\ttrain loss: %0.4f" % score
        print "\ttrain accuracy: %0.4f" % acc

        print 'Evaluate test data...'
        score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
        print "\ttest loss: %0.4f" % score
        print "\ttest accuracy: %0.4f" % acc

def evaluate_posterior(setting):
    print '\nEvaluating posterior...'
    datapath = variables.DATAPATH
    categories = variables.CATEGORIES
    batch_size = variables.BATCH_SIZE

    for category in categories:
        modelpath = datapath + category + '/v4/keras_' + setting + '/'

        model = model_from_json(open(modelpath + 'keras_model_architecture.json').read())
        model.load_weights(modelpath + 'keras_model_weights.h5')

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


        print '\ncategory: ' + category
        current_reviews_train = np.load(datapath + category + '/hdl_reviews_shuffled_embeddedalla.npy')
        current_reviews_test = np.load(datapath + category + '/hdl_reviews_shuffled_embeddedallb.npy')

        raw_ratings_train = np.genfromtxt(datapath + category + '/ratings_shuffleda')
        raw_ratings_test = np.genfromtxt(datapath + category + '/ratings_shuffledb')

        current_ratings_train = np.zeros(raw_ratings_train.shape)
        current_ratings_train[raw_ratings_train > 3] = 1
        current_ratings_train[raw_ratings_train < 3] = 0
        current_ratings_test = np.zeros(raw_ratings_test.shape)
        current_ratings_test[raw_ratings_test > 3] = 1
        current_ratings_test[raw_ratings_test < 3] = 0

        del raw_ratings_train, raw_ratings_test

        X_train = current_reviews_train
        X_test = current_reviews_test
        y_train = current_ratings_train
        y_test = current_ratings_test

        del current_reviews_train, current_reviews_test
        del current_ratings_train, current_ratings_test

        print 'Evaluate training data...'
        score, acc = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
        print "\ttrain loss: %0.4f" % score
        print "\ttrain accuracy: %0.4f" % acc

        print 'Evaluate test data...'
        score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
        print "\ttest loss: %0.4f" % score
        print "\ttest accuracy: %0.4f" % acc

if __name__ == '__main__':
    setting = 'globalEmbeddings_traintestsplit_shuffled_v2'

    evaluate_prior(setting)
    evaluate_posterior(setting)
