# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:04:42 2015

@author: ssc317
"""
import gzip
from collections import defaultdict
import mylib
def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

data_ = []
for l in readGz("../train.json.gz"):
    data_.append(l)
train_data = data_[:500000]
valid_data = data_[500000:]
mylib.saveData('../train_valid_1M',[train_data, valid_data])
mylib.saveData('../train_1M',[data_])