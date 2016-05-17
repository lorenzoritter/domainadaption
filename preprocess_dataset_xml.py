#!/usr/bin/env python

'''preprocess_dataset_xml.py

Preprocessing of the all.review file in the Linux shell:
1. Add <data> </data> tags around file
    sed -i -e '1i<data>\' all.review
    echo '</data>' >> all.review
2. Delete <reviewer_location> field (leads to parsing errors and is unneccessary)
    sed -e "/<reviewer_location>/,/<\/reviewer_location>/d" -i all.review
3. Delete <br> tags
    sed 's/<br>/ /g' -i all.review
    sed 's/<BR>/ /g' -i all.review
'''

__author__ = "Lorenzo von Ritter"
__date__ = "2016-05-17"

#!/usr/bin/env python

'''preprocess_dataset.py

file description here
'''

__author__ = "Lorenzo von Ritter"
__date__ = "2016 - 04 - 29"

import csv
import os
import helper
import review_tokenizer
import variables
import xml.etree.ElementTree
import lxml.etree


filepath = variables.DATAPATH
categories = variables.CATEGORIES
sentiments = variables.SENTIMENTS
include_title = False

for category in categories:
    print 'category: ' + category
    for sentiment in sentiments:
        # read raw data and make it readable using BeautifulSoup
        sourcefile = filepath + category + '/' + sentiment + '2.review'
        data = open(sourcefile).read()

        parser = lxml.etree.XMLParser(encoding='ISO-8859-1', recover=True)
        readable_data = xml.etree.ElementTree.parse(sourcefile, parser=parser).getroot()

        # set up target files
        targetpath = filepath + category + '/'
        if not os.path.exists(targetpath):
            os.makedirs(targetpath)

        # remove previous results
        reviewfile = targetpath + 'reviews_' + sentiment + '.txt'
        if os.path.exists(reviewfile):
            os.remove(reviewfile)

        ratingfile = targetpath + 'ratings_' + sentiment + '.txt'
        if os.path.exists(ratingfile):
            os.remove(ratingfile)

        # preprocess data and write to csv files
        with open(reviewfile, 'wb') as reviewcsv:
            reviewswriter = csv.writer(reviewcsv, quoting=csv.QUOTE_NONE, delimiter='|', escapechar='\\')

            with open(ratingfile, 'wb') as ratingcsv:
                ratingswriter = csv.writer(ratingcsv)

                reviewcount = 0
                successcount = 0

                #iterable = readable_data.find_all('review')
                iterable = readable_data.findall('review')

                #for review in iterable:
                for review in iterable:

                    reviewcount += 1

                    # write review to file
                    try:
                        review_text = review.find('review_text').text
                        review_text = helper.clean(review_text) # remove newlines, quotation marks, unicode
                        review_text = review_tokenizer.cleanOnereview(review_text, removesinglewords=False) # tokenize into a list of sentences
                        review_text = " <eos> ".join(review_text) # join with <eos> tags
                        # TODO: evaluate sentiment classification performance when using <eos> tag

                        if include_title:
                            review_title = review.find('title').text
                            review_title = helper.clean(review_title)
                            review_title = review_tokenizer.cleanOnereview(review_title, removesinglewords=False)
                            review_title = " ".join(review_title)

                            reviewswriter.writerow([review_title[:] + ' <eos> ' + review_text[:]])

                        else:
                            reviewswriter.writerow([review_text[:]])

                        # write rating to file
                        rating = review.find('rating').text
                        rating = helper.clean(rating)
                        ratingswriter.writerow([rating])

                        successcount += 1

                    except:
                        print '\tNot able to read review ' + str(reviewcount) + ' with ID ' + \
                              review.find('unique_id').text.strip()[:40]
        print '\tread %d reviews, saved %d successfully.' %(reviewcount, successcount)