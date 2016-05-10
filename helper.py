#!/usr/bin/env python

'''helper.py

file description here
'''

__author__ = "Lorenzo von Ritter"
__date__ = "2016 - 05 - 03"


def clean(input):
    # remove leading and tailing white space
    input = input.strip()

    # replace single and double newlines with a space
    input = input.replace('\n\n', ' ').replace('\n', ' ')

    # remove quotation marks
    input = input.replace('&quot;', '').replace('"','')

    # remove unicode
    input = input.encode('ascii', 'ignore')

    # replace list by enumeration
    input = input.replace(' *',', ')

    # make text lower case
    input = input.lower()

    return input