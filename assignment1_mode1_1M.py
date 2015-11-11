# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:35:40 2015

@author: ssc317
"""

feature_word = ['book',
 'one',
 'story',
 'read',
 'like',
 'love',
 'would',
 'characters',
 'really',
 'time']
feature_word = []
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:02:00 2015

@author: ssc317
"""

import numpy
import string
import mylib
[train_data, valid_data] = mylib.loadData('../train_valid_1M')
td = []
i = 0
num = 100000
for t in train_data:
    if t['helpful']['outOf'] != 0:
        td.append(t)
    i += 1
    if i == num:
        break
train_data = td
helpful_train_y = [-1 if t['helpful']['outOf'] == 0 else t['helpful']['nHelpful'] * 1.0 /t['helpful']['outOf'] for t in train_data]


def feature(datum):
  feat = [1]
  punc = string.punctuation
  review = ''.join([o.lower() if not o in punc else ' ' for o in list(datum['reviewText'])]).split()
  feat.append(len(review))
  feat.append(datum['rating'])
  for word in feature_word:
      feat.append(review.count(word))
  return feat
  
helpful_train_X = [feature(d) for d in train_data]
theta,residuals,rank,s = numpy.linalg.lstsq(helpful_train_X, helpful_train_y)
mylib.saveData('../data/theta_mode1_01M',[theta])

# In[] validation
import numpy
import string
import mylib
from copy import deepcopy
valid_data = valid_data[-100000:]
vd = []
for v in valid_data:
    if t['helpful']['outOf'] != 0:
        vd.append(v)
valid_data = vd
helpful_valid_y = [0 if t['helpful']['outOf'] == 0 else t['helpful']['nHelpful'] * 1.0 /t['helpful']['outOf'] for t in valid_data]

def feature(datum):
  feat = [1]
  punc = string.punctuation
  review = ''.join([o.lower() if not o in punc else ' ' for o in list(datum['reviewText'])]).split()
  feat.append(len(review))
  feat.append(datum['rating'])
  for word in feature_word:
      feat.append(review.count(word))
  return feat
  

helpful_valid_X = [feature(d) for d in valid_data]
helpful_valid_predict = [sum([a*b for a,b in zip(x, theta)]) for x in helpful_valid_X]
helpful_valid_MAE = sum([abs(a-b) for a,b in zip(helpful_valid_y, helpful_valid_predict)]) / len(helpful_valid_y)

print "validation MAE on nHelpful/outof is " + str(helpful_valid_MAE)



# In[] predict
import numpy
import string
import mylib
from copy import deepcopy
pairs_Helpful = []
pairs_HelpfulOf0 = []
f = open('./pairs_Helpful.txt')
for line in f:
    if line.startswith('userID'):
        pass
    elif line.startswith('U'):
        UIO = line.split('-')
        UIO[2] = int(UIO[2])
        if UIO[2] != 0:
            pairs_Helpful.append(UIO)
        else:
            pairs_HelpfulOf0.append(UIO)
pairs_Helpful0 = deepcopy(pairs_Helpful)

pairs_Helpful.sort(key=lambda x:x[0])
pairs_Helpful.sort(key=lambda x:x[1])

data_ = mylib.parseDataFromFile("../helpful.json")
data_.sort(key=lambda x:x['reviewerID'])
data_.sort(key=lambda x:x['itemID'])
test_data = []
findNum = 0
index = 0
for p in pairs_Helpful:
    for index in range(index, len(data_)):
        d = data_[index]
        if d['itemID'] == p[1] and d['reviewerID'] == p[0]:
            findNum += 1
            test_data.append(d)
            break
        

helpful_test_X = [feature(d) for d in test_data]
helpful_test_predict = [sum([a*b for a,b in zip(x, theta)]) for x in helpful_test_X]
helpful_test_result = [[t['reviewerID']+'-'+t['itemID']+'-'+str(t['helpful']['outOf']),round(d*t['helpful']['outOf'])] for t,d in zip(test_data, helpful_test_predict)]
helpful_test_resultOf0 = [[t[0] +'-'+t[1]+'-'+str(t[2]),0] for t in pairs_HelpfulOf0]
helpful_test_result += helpful_test_resultOf0


# In[] generate csv file
import mylib
import csv
def saveCSV(filename, data):
    f =  open(filename, 'wb') 
    writer = csv.writer(f)
    writer.writerow(['userID-itemID-outOf', 'prediction'])
    for d in data:
        writer.writerow(d)
    f.close()
    
saveCSV('helpful_test_result.csv',helpful_test_result)
