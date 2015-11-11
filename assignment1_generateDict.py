# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:21:43 2015

@author: ssc317
"""
# In[] generate small train file
import gzip
from collections import defaultdict

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)
    

import string
import mylib
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

dict_all, dict_count, dict_nHelpful, dict_helpful = defaultdict(float),defaultdict(int),defaultdict(float),defaultdict(float)

i = 0

for l in readGz("../train.json.gz"):
    outOf = l['helpful']['outOf']
    review = ''.join([o for o in list(l['reviewText']) if not o in punc]).split()
    for word in review:
        if word not in stopwords:
            if outOf != 0:
                nHelpful = l['helpful']['nHelpful'] * 1.0;
                dict_count[word] += 1
                dict_nHelpful[word] += nHelpful
                dict_helpful[word] += nHelpful / outOf
            dict_all[word] += 1
    i += 1
    if i == 500:
        mylib.saveData('../dicts_500',[dict_all, dict_count, dict_nHelpful, dict_helpful])
    elif i == 5000:
        mylib.saveData('../dicts_5000',[dict_all, dict_count, dict_nHelpful, dict_helpful])
    elif i == 50000:
        mylib.saveData('../dicts_50000',[dict_all, dict_count, dict_nHelpful, dict_helpful])
    elif i == 500000:
        mylib.saveData('../dicts_500000',[dict_all, dict_count, dict_nHelpful, dict_helpful])
    if i > 500000:
        break