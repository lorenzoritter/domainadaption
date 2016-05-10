#!/usr/bin/env python

'''preprocess_dataset_for_embeddings.py

file description here
'''

__author__ = "Lorenzo von Ritter"
__date__ = "2016 - 05 - 09"

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

    # set up target files
    workingpath = filepath + category + '/'
    if not os.path.exists(workingpath):
        os.makedirs(workingpath)

    # remove previous results
    reviewfile = workingpath + 'reviews_forEmbedding.txt'
    if os.path.exists(reviewfile):
        os.remove(reviewfile)

    # preprocess data and write to csv files
    with open(reviewfile, 'wb') as reviewcsv:
        reviewswriter = csv.writer(reviewcsv, quoting=csv.QUOTE_NONE, delimiter='|', escapechar='\\')

        for sentiment in ['positive', 'negative', 'unlabeled']:
            print '\t' + sentiment
            # read raw data and make it readable using BeautifulSoup
            sourcefile = workingpath + sentiment + '.review'
            data = open(sourcefile).read()
            readable_data = BeautifulSoup(data, 'html.parser')

            reviewcount = 0

            for review in readable_data.find_all('review'):

                reviewcount += 1

                # write review title to file
                if include_title:
                    try:
                        review_title = review.title.string
                        review_title = helper.clean(review_title)
                        review_title = review_tokenizer.cleanOnereview(review_title)

                        for sentence in review_title:
                            reviewswriter.writerow([sentence])
                    except:
                        print '\tNot able to read ' + sentiment + ' review title' + str(reviewcount) + ' with ID ' + \
                              review.unique_id.string.strip()[:40]

                # write review to file
                try:
                    review_text = review.review_text.string
                    review_text = helper.clean(review_text)  # remove newlines, quotation marks, unicode
                    review_text = review_tokenizer.cleanOnereview(review_text)  # tokenize into a list of sentences

                    for sentence in review_text:
                        reviewswriter.writerow([sentence])
                except:
                    print '\t\ŧNot able to read ' + sentiment + ' review ' + str(reviewcount) + ' with ID ' + \
                          review.unique_id.string.strip()[:40]
