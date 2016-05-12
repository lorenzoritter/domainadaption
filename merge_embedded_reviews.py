#!/usr/bin/env python

'''merge_embedded_reviews.py

file description here
'''

__author__ = "Lorenzo von Ritter"
__date__ = "2016-05-09"

import numpy as np

categories = ['books', 'dvd', 'electronics', 'kitchen_&_housewares']
sentiments = ['positive', 'negative']

filepath = '/home/lorenzo/PycharmProjects/domainadaption/sorted_data_acl/'


for sentiment in sentiments:
    print 'sentiment: ' + sentiment
    flag = 0

    for category in categories:
        print '\tcategory: ' + category
        file = np.load(filepath + category + '/reviews_' + sentiment + '.npy')

        if flag==0:
            outfile = file
            flag = 1

        else:
            outfile = np.concatenate((outfile, file))

    np.save(filepath + 'all/reviews_' + sentiment + '.npy', outfile)