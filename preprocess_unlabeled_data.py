from bs4 import BeautifulSoup
import csv
import os

categories = ['books', 'dvd', 'electronics', 'kitchen_&_housewares']

filepath = '/home/lorenzo/PycharmProjects/domainadaption/sorted_data_acl/'

include_title = True

for category in categories:
    print 'category: ' + category
    for sentiment in ['unlabeled']:
        # read raw data and make it readable using BeautifulSoup
        sourcefile = filepath + category + '/' + sentiment + '.review'
        data = open(sourcefile).read()
        readable_data = BeautifulSoup(data, 'html.parser')

        # set up target files
        targetpath = filepath + category + '/'
        if not os.path.exists(targetpath):
            os.makedirs(targetpath)

        # remove previous results
        reviewfile = targetpath + sentiment + '_reviews.csv'
        if os.path.exists(reviewfile):
            os.remove(reviewfile)


        # preprocess data and write to csv files
        with open(reviewfile, 'wb') as reviewcsv:
            reviewswriter = csv.writer(reviewcsv, delimiter='.')

            reviewcount = 0

            for review in readable_data.find_all('review'):

                reviewcount += 1

                # write review to file
                try:
                    review_text = review.review_text.string.strip().replace('\n\n', ' ').replace('\n', ' ')
                    review_text = review_text.encode('utf8', 'ignore')

                    if include_title:
                        review_title = review.review_text.string.strip().replace('\n\n', ' ').replace('\n', ' ')
                        review_title = review_title.encode('utf8', 'ignore')
                        reviewswriter.writerow([review_text + '. ' + review_title])
                    else:
                        reviewswriter.writerow([review_text])
                except:
                    print 'Not able to read review ' + str(reviewcount)
