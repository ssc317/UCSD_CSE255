# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:07:04 2015

@author: ssc317
"""

import gzip
import mylib
from collections import defaultdict
import string
def readGz(f):
  for l in gzip.open(f):
    yield eval(l)
data = []
for l in readGz("../train.json.gz"):
    data.append(l)
mylib.saveData('../1Mtrain', [data])