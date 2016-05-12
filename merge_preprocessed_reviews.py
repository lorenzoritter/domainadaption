#!/usr/bin/env python

'''merge_preprocessed_reviews.py

Merge the preprocessed reviews from the four categories into concatenated files in the all/ folder'''

__author__ = "Lorenzo von Ritter"
__date__ = "2016-05-12"

categories = ['books', 'dvd', 'electronics', 'kitchen_&_housewares']
sentiments = ['positive', 'negative']

filepath = '/home/lorenzo/PycharmProjects/domainadaption/sorted_data_acl/'


for sentiment in sentiments:
    print 'sentiment: ' + sentiment

    with open(filepath + 'all/reviews_' + sentiment + '.txt', 'w') as outfile:

        for category in categories:
            print '\tcategory: ' + category

            with open(filepath + category + '/reviews_' + sentiment + '.txt') as infile:
                outfile.write(infile.read())