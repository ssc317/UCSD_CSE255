# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 22:56:19 2015

@author: ssc317
"""

import mylib
import csv
filename = '1M_train_rating'
pairs_Rating = []
f = open('./pairs_Rating.txt')
for line in f:
    if line.startswith('userID'):
        pass
    elif line.startswith('U'):
        UIO = line.split('-')
        pairs_Rating.append(UIO)
[alpha, beta_u, beta_i] = mylib.loadData(filename)
rating_test_predict = [alpha+beta_u[pr[0]]+beta_i[pr[1]] for pr in pairs_Rating]
rating_test_result = [[pr[0]+'-'+pr[1].rstrip(), str(ptr)] for pr,ptr in zip(pairs_Rating,rating_test_predict)]


def saveCSV(filename, data):
    f =  open(filename, 'wb') 
    writer = csv.writer(f)
    writer.writerow(['userID-itemID', 'prediction'])
    for d in data:
        writer.writerow(d)
    f.close()
saveCSV('rating_test_result.csv',rating_test_result)
