
import os
import re
import string
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

def cleanOnereview(onereview, removesinglewords=True):
    # tokenize sentences and words
    sentlist = sent_tokenize(onereview)
    tokenized_sentlist = [word_tokenize(doc) for doc in sentlist]

    # remove puncatuation
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    sentlist_no_punctuation = []
    for sent in tokenized_sentlist:    
        new_sent = []
        for token in sent: 
            new_token = regex.sub('', token)
            if not new_token == '':
                new_sent.append(new_token)
        if removesinglewords and len(new_sent)<=1:   # Remove the sentence with only one word (only for word embedding).
            pass
        else:
            tem = ' '.join(new_sent)        # + '\n'    # removed the newline
            sentlist_no_punctuation.append(tem)

    return sentlist_no_punctuation

# onereview = 'hello world! Yeep! I *like* the movie.'
# print cleanOnereview(onereview)
 


def cleanReviewFile(sourcefile,targetfile):
    # Remove previous results
    if os.path.exists(targetfile):
        os.remove(targetfile)

    # Clean reviews and write to a new file
    with open(sourcefile,'r') as s, open(targetfile,'w') as w:
        bunchsize = 5000
        bunch = []
        for onereview in s:
            cleaned_sent_list = cleanOnereview(onereview)

            bunch.extend(cleaned_sent_list)
            if len(bunch) == bunchsize:
                w.writelines(bunch)
                bunch = []
        w.writelines(bunch)
    
# sourcefile = 'example_reviews.csv'
# targetfile = 'example_reviews_cleaned.csv'
# cleanReviewFile(sourcefile,targetfile)
# tem = open(targetfile,'r').readlines()
# print len(tem)



def concatReviewFiles(reviewfilenames,targetfile):
    if os.path.exists(targetfile):
        os.remove(targetfile)
    with open(targetfile, 'w') as outfile:
        for fname in reviewfilenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

# reviewfilenames = ['example_reviews.csv','example_reviews.csv','example_reviews.csv']
# targetfile = 'example_reviews_cleaned_comb.csv'
# concatReviewFiles(reviewfilenames,targetfile)
# tem = open(targetfile,'r').readlines()
# print len(tem)



def run():
    filepath = '/home/zhaoxu/venv/sentiment/data/amazon/acl2007/sorted_data_acl/'
    categories = ['books', 'dvd', 'electronics', 'kitchen_&_housewares']
    sentiment = 'all'
    
    for category in categories:
        sourcefile = filepath + category + '/' + sentiment + '_reviews.csv'
        targetfile = filepath + category + '/' + sentiment + '_reviews_cleaned.csv'
        cleanReviewFile(sourcefile,targetfile)


#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
#run()