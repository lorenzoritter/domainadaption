#!/usr/bin/env python

'''count_wordsNotInVocab.py

file description here
'''

from __future__ import division
import numpy as np

__author__ = "Lorenzo von Ritter"
__date__ = "2016-05-24"

datapath = '/home/lorenzo/PycharmProjects/domainadaption/sorted_data_acl/'
categories = ['books']

for category in categories:
    print 'category: ' + category

    reviews_positive = np.load(datapath + category + '/hdl_reviews_positive_embeddedall.npy')
    reviews_negative = np.load(datapath + category + '/hdl_reviews_negative_embeddedall.npy')
    reviews = np.append(reviews_positive, reviews_negative, axis=0)
    del reviews_positive, reviews_negative

    simple_reviews = np.dot(reviews, np.ones((250,1)))

    words = 0
    zeros = 0

    for line in simple_reviews:
        tmp = np.trim_zeros(line, 'b')
        words = words + len(tmp)
        zeros = zeros + len(tmp) - np.count_nonzero(tmp)

    print '\tzeros in small dataset: %0.4f' %(zeros/words)


    reviews_samples = np.load(datapath + category + '/hdl_reviews_all_embeddedall_samples.npy')

    simple_reviews_samples = np.dot(reviews_samples, np.ones((250, 1)))

    words = 0
    zeros = 0

    for line in simple_reviews_samples:
        tmp = np.trim_zeros(line, 'b')
        words = words + len(tmp)
        zeros = zeros + len(tmp) - np.count_nonzero(tmp)

    print '\tzeros in large dataset: %0.4f' %(zeros/words)