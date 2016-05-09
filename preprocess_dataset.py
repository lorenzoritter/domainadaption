#!/usr/bin/env python

'''preprocess_dataset.py

file description here
'''

__author__ = "Lorenzo von Ritter"
__date__ = 2016 - 04 - 29

from bs4 import BeautifulSoup
import csv
import os
import helper
import review_tokenizer


categories = ['books', 'dvd', 'electronics', 'kitchen_&_housewares']

filepath = '/home/lorenzo/PycharmProjects/domainadaption/sorted_data_acl/'

include_title = True

for category in categories:
    print 'category: ' + category
    for sentiment in ['positive', 'negative']:
        # read raw data and make it readable using BeautifulSoup
        sourcefile = filepath + category + '/' + sentiment + '.review'
        data = open(sourcefile).read()
        readable_data = BeautifulSoup(data, 'html.parser')

        # set up target files
        targetpath = filepath + category + '/'
        if not os.path.exists(targetpath):
            os.makedirs(targetpath)

        # remove previous results
        reviewfile = targetpath + sentiment + '_reviews.txt'
        if os.path.exists(reviewfile):
            os.remove(reviewfile)

        ratingfile = targetpath + sentiment + '_ratings.txt'
        if os.path.exists(ratingfile):
            os.remove(ratingfile)

        # preprocess data and write to csv files
        with open(reviewfile, 'wb') as reviewcsv:
            reviewswriter = csv.writer(reviewcsv, quoting=csv.QUOTE_NONE, delimiter='|', escapechar='\\')

            with open(ratingfile, 'wb') as ratingcsv:
                ratingswriter = csv.writer(ratingcsv)

                reviewcount = 0

                for review in readable_data.find_all('review'):

                    reviewcount += 1

                    # write review to file
                    try:
                        review_text = review.review_text.string
                        review_text = helper.clean(review_text) # remove newlines, quotation marks, unicode
                        review_text = review_tokenizer.cleanOnereview(review_text) # tokenize into a list of sentences
                        review_text = " ".join(review_text) # join with whitespaces

                        if include_title:
                            review_title = review.review_text.string
                            review_title = helper.clean(review_title)
                            review_title = review_tokenizer.cleanOnereview(review_title)
                            review_title = " ".join(review_title)

                            reviewswriter.writerow([review_text[:] + '. ' + review_title[:]])

                        else:
                            reviewswriter.writerow([review_text[:]])

                        # write rating to file
                        rating = review.rating.string
                        rating = helper.clean(rating)
                        ratingswriter.writerow([rating])
                    except:
                        print '\tNot able to read review ' + str(reviewcount) + ' with ID ' + \
                              review.unique_id.string.strip()[:40]
