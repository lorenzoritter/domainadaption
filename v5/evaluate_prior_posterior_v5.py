#!/usr/bin/env python

'''evaluate_prior_posterior_v5.py

file description here
'''

__author__ = "Lorenzo von Ritter"
__date__ = "2016-06-02"

import numpy as np
import pandas as pd
from keras.models import model_from_json
import variables_v5 as variables


def evaluate_prior(setting, verbose=1):
    print 'Evaluating prior...'
    datapath = variables.DATAPATH
    categories = variables.CATEGORIES
    batch_size = variables.BATCH_SIZE

    # create table for results
    results = pd.DataFrame(index=['train_loss', 'train_acc', 'test_loss', 'test_acc'], columns=categories)

    # load prior. since it is a global model, it only need to be loaded once
    modelpath = datapath + 'all/v5/keras_' + setting + '/'

    model = model_from_json(open(modelpath + 'keras_model_architecture.json').read())
    model.load_weights(modelpath + 'keras_model_weights.h5')

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    for category in categories:
        if verbose:
            print 'category: %s' % category

        # read train and test reviews (words will be represented as integers)
        current_reviews_train = np.load(datapath + category + '/reviews_shuffled_embeddedIDa.npy')
        current_reviews_test = np.load(datapath + category + '/reviews_shuffled_embeddedIDb.npy')

        # read train and test ratings (1, 2, 4 or 5 stars)
        raw_ratings_train = np.genfromtxt(datapath + category + '/ratings_shuffleda')
        raw_ratings_test = np.genfromtxt(datapath + category + '/ratings_shuffledb')

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

        del raw_ratings_train, raw_ratings_test

        X_train = current_reviews_train
        X_test = current_reviews_test
        y_train = current_ratings_train
        y_test = current_ratings_test

        del current_reviews_train, current_reviews_test
        del current_ratings_train, current_ratings_test

        if verbose:
            print 'Evaluate training data...'
        score, acc = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
        results[category]['train_loss'] = '{0:.4f}'.format(score)
        results[category]['train_acc'] = '{0:.4f}'.format(acc)
        # print "\ttrain loss: %0.4f" % score
        # print "\ttrain accuracy: %0.4f" % acc

        if verbose:
            print 'Evaluate test data...'
        score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
        results[category]['test_loss'] = '{0:.4f}'.format(score)
        results[category]['test_acc'] = '{0:.4f}'.format(acc)
        # print "\ttest loss: %0.4f" % score
        # print "\ttest accuracy: %0.4f" % acc

    return results

def evaluate_posterior(setting, verbose=1):
    print 'Evaluating posterior...'
    datapath = variables.DATAPATH
    categories = variables.CATEGORIES
    batch_size = variables.BATCH_SIZE

    # create table for results
    results = pd.DataFrame(index=['train_loss', 'train_acc', 'test_loss', 'test_acc'], columns=categories)

    for category in categories:
        if verbose:
            print '\ncategory: %s' % category
        # load posterior. each category has a different posterior
        modelpath = datapath + category + '/v5/keras_' + setting + '/'

        model = model_from_json(open(modelpath + 'keras_model_architecture.json').read())
        model.load_weights(modelpath + 'keras_model_weights.h5')

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        # read train and test reviews (words will be represented as integers)
        reviews_train = np.load(datapath + category + '/reviews_shuffled_embeddedIDa.npy')
        reviews_test = np.load(datapath + category + '/reviews_shuffled_embeddedIDb.npy')

        # read train and test ratings (1, 2, 4 or 5 stars)
        raw_ratings_train = np.genfromtxt(datapath + category + '/ratings_shuffleda')
        raw_ratings_test = np.genfromtxt(datapath + category + '/ratings_shuffledb')

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

        if verbose:
            print 'Evaluate training data...'
        score, acc = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
        results[category]['train_loss'] = '{0:.4f}'.format(score)
        results[category]['train_acc'] = '{0:.4f}'.format(acc)
        # print "\ttrain loss: %0.4f" % score
        # print "\ttrain accuracy: %0.4f" % acc

        if verbose:
            print 'Evaluate test data...'
        score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
        results[category]['test_loss'] = '{0:.4f}'.format(score)
        results[category]['test_acc'] = '{0:.4f}'.format(acc)
        # print "\ttest loss: %0.4f" % score
        # print "\ttest accuracy: %0.4f" % acc

    return results

if __name__ == '__main__':
    setting_prior = ''
    setting_posterior = ''

    evaluate_prior(setting_prior)
    evaluate_posterior(setting_posterior)
