#!/usr/bin/env python

'''variables.py

file description here
'''

__author__ = "Lorenzo von Ritter"
__date__ = "2016-05-11"

import os

EMBEDDING_DIM = 250
MAX_LENGTH = 200

os.environ['EMBEDDING_DIM'] = str(EMBEDDING_DIM)
os.environ['MAX_LENGTH'] = str(MAX_LENGTH)