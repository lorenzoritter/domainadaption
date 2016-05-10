#!/usr/bin/env python

'''review_tokenizer.py

file description here
'''

__author__ = "Zhao Xu"
__date__ = "2016 - 05 - 09"


import re
import string
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

def cleanOnereview(onereview):
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
        if len(new_sent)>1:   # Remove the sentence with only one word (only for word embedding).
            tem = ' '.join(new_sent)
            sentlist_no_punctuation.append(tem)

    return sentlist_no_punctuation

#onereview = 'hello world! Yeep! I *like* the movie.'
#s = cleanOnereview(onereview)
#print s