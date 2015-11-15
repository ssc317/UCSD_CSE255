# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 17:36:33 2015

@author: ssc317
"""
from collections import defaultdict
from math import exp
import numpy

# In[]================================================================================
# In[] Rui, Iu, Ui
def getRui(data):
    Rui = {};
    for d in data:
        Rui[(d['reviewerID'], d['itemID'])] = d['rating']
    return Rui
def getIu_Ui(data):
    Iu, Ui = {},{}
    for d in data:
        if d['reviewerID'] not in Iu:
            Iu[d['reviewerID']]= []
        if d['itemID'] not in Ui:
            Ui[d['itemID']] = []
        Iu[d['reviewerID']].append(d['itemID'])
        Ui[d['itemID']].append(d['reviewerID'])
    return [Iu, Ui]
# In[]================================================================================
# In[] MSE
def mse(rating_valid_predict, rating_valid_y):
        return sum((a-b)**2 for a,b in zip(rating_valid_predict,rating_valid_y)) / len(rating_valid_y)
# In[]================================================================================
# In[] Dirty function definition
# In[] find dirty data
from collections import defaultdict
import numpy
def getDirtyUI(Rui, Iu, Ui, dirty_feature):
    dirty_u = {}
    dirty_i = {}
    [dirty_limit, rating_std, rating_bound] = dirty_feature
    for u in Iu.keys():
        if dirty_limit < len(Iu[u]):
            rating_list = []
            for i in Iu[u]:
                rating_list.append(Rui[(u,i)])
            print "test1" + str(rating_list)
	    print "test2" + str(numpy.std(rating_list))
            if numpy.std(rating_list) < rating_std:
                if numpy.mean(rating_list)>rating_bound[1] or numpy.mean(rating_list)<rating_bound[0]:
                    dirty_u[u] = numpy.mean(rating_list)
    for i in Ui.keys():
        if dirty_limit < len(Ui[i]):
            rating_set = set()
            for u in Ui[i]:
                rating_set.add(Rui[(u,i)])
            if len(rating_set) == 1:
                dirty_i[i] = list(rating_set)[0]
    return [dirty_u, dirty_i]
# In[]================================================================================
# In[] convergence for dirty
from collections import defaultdict
def convergence_dirty(train_data,lamda, dirty_feature):
    Rui = getRui(train_data)
    [Iu, Ui] = getIu_Ui(train_data)
    [dirty_u, dirty_i] = getDirtyUI(Rui, Iu, Ui, dirty_feature)
    train_data_without_dirty = []
    for t in train_data:
        if t['reviewerID'] not in dirty_u.keys() and t['itemID'] not in dirty_i.keys():
            train_data_without_dirty.append(t)
    Rui = getRui(train_data_without_dirty)
    [Iu, Ui] = getIu_Ui(train_data_without_dirty)
    alpha, beta_u, beta_i, iterNum, convariance = 0, { uid: 0 for uid in Iu.keys()}, { iid: 0 for iid in Ui.keys()}, 0, 0
    while iterNum > -1:
        new_alpha = sum([Rui[(t['reviewerID'], t['itemID'])] - beta_u[t['reviewerID']] - beta_i[t['itemID']] for t in train_data_without_dirty]) / len(train_data_without_dirty)
        new_beta_i, new_beta_u = defaultdict(int), defaultdict(int)
        for u in Iu.keys():
            new_beta_u[u] = sum([ Rui[(u,i)] - new_alpha - beta_i[i] for i in Iu[u]]) / (lamda + len(Iu[u]))
        for i in Ui.keys():
            new_beta_i[i] = sum([ Rui[(u,i)] - alpha - new_beta_u[u] for u in Ui[i]]) / (lamda + len(Ui[i]))
        new_convariance = [(alpha - new_alpha)**2] + [(b-nb)**2 for b,nb in zip(beta_i.values(), new_beta_i.values())] + [(b-nb)**2 for b,nb in zip(beta_u.values(), new_beta_u.values())]
        new_convariance = sum(numpy.sqrt(new_convariance))
        print "covariance is "+str(new_convariance)
        #if(alpha==new_alpha and beta_i == new_beta_i and beta_u == new_beta_u):
        if(new_convariance < 1e-10 or convariance == new_convariance):
            break
        else:
            alpha, beta_i, beta_u, iterNum, convariance = new_alpha, new_beta_i, new_beta_u, iterNum+1, new_convariance
            print "Finish iter " + str(iterNum) + "with lamda " + str(lamda)
    return [alpha, beta_i, beta_u, dirty_u, dirty_i]
# In[]================================================================================
# In[] validate for dirty  
# ?????????how to deal with new?????????
def validate_dirty(valid_data, dirty_u, dirty_i, alpha, beta_u, beta_i):
    rating_valid_parameters = []
    for d in valid_data:
        if d['reviewerID'] in dirty_u.keys():
            rating_valid_parameters.append([dirty_u[d['reviewerID']],0,0])
        elif d['itemID'] in dirty_i.keys():
            rating_valid_parameters.append([dirty_i[d['itemID']],0,0])
        else:
            rating_valid_parameters.append([alpha, beta_u[d['reviewerID']], beta_i[d['itemID']]])
    rating_valid_predict = [sum(para) for para in rating_valid_parameters]
    rating_valid_y = [d['rating'] for d in valid_data]
    rating_valid_MSE = mse(rating_valid_predict, rating_valid_y)
    print "MSE of validation set is "+ str(rating_valid_MSE)
    return rating_valid_MSE
# In[]================================================================================
# In[] test dirty  
import mylib
import csv
def test_dirty(alpha, beta_u, beta_i, dirty_u, dirty_i):
    pairs_Rating = []
    f = open('./data/pairs_Rating.txt')
    for line in f:
        if line.startswith('userID'):
            pass
        elif line.startswith('U'):
            UIO = line.split('-')
            UIO[1] = UIO[1].rstrip()
            pairs_Rating.append(UIO)
    rating_test_predict = []
    for pr in pairs_Rating:
        if pr[0] in dirty_u.keys():
            rating_test_predict.append(dirty_u[pr[0]])
        elif pr[1] in dirty_i.keys():
            rating_test_predict.append(dirty_i[pr[1]])
        else:
            rating_test_predict.append(alpha+beta_u[pr[0]]+beta_i[pr[1]])
    rating_test_result = [[pr[0]+'-'+pr[1], str(ptr)] for pr,ptr in zip(pairs_Rating,rating_test_predict)]
    
    saveCSV('rating_test_result.csv',rating_test_result)
    
# In[]================================================================================
# In[] main
[data_] = mylib.loadData('../1Mtrain')
train_data = data_[:900000]
valid_data = data_[900000:]
del data_

dirty_limits = [10,11,12,13,14,15,16,17,18,19,20]
sds = [0.3,0.5,0.7,0.9,1.0,2.0]
dirty_bound = [[2.6,4.4],[2.8,4.2],[3,4],[3.5,3.5]]
MSEs = [[[0 for dbi in range(len(dirty_bound))] for si in range(len(sds))] for dli in range(len(dirty_limits))]
thetas = [[[0 for dbi in range(len(dirty_bound))] for si in range(len(sds))] for dli in range(len(dirty_limits))]
for dli in range(len(dirty_limits)):
    dl = dirty_limits[dli]
    for si in range(len(sds)):
        s = sds[si]
        for dbi in range(len(dirty_bound)):
            db = dirty_bound[dbi]
            dirty_feature = [dl, db, s]
            [alpha, beta_i, beta_u, dirty_u, dirty_i] = convergence_dirty(train_data, 3, dirty_feature)
            MSE = validate_dirty(valid_data, dirty_u, dirty_i, alpha, beta_u, beta_i)
            MSEs[dli][si][dbi] = MSE
            thetas[dli][si][dbi] = [alpha, beta_i, beta_u, dirty_u, dirty_i]
            print "dirty_limit: " + str(dl) + "  std: "+str(s)+ "  dirty bound: "+ str(db) + "==> MSE: " + MSE 
            mylib.saveData('dirty_features_theta', [MSEs,thetas])
print MSEs
