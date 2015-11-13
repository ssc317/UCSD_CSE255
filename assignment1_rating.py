# In[] load 1M data
import mylib
#data_ = mylib.loadData('../1Mtrain')
#train_data = data_[:900000]
#valid_data = data_[900000:]
[train_data, valid_data] = mylib.loadData('../HW3train_valid')
# In[] calculate with a lamda
from collections import defaultdict
from math import exp
import numpy

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

Rui = getRui(train_data)
[Iu, Ui] = getIu_Ui(train_data)

alpha0 = 4.21824 #numpy.random.rand() #4.21824
beta_u0 = { uid: 0 for uid in Iu.keys()}
beta_i0 = { iid: 0 for iid in Ui.keys()}
rating_valid_y = [d['rating'] for d in valid_data]
def mse(rating_valid_predict, rating_valid_y):
        return sum((a-b)**2 for a,b in zip(rating_valid_predict,rating_valid_y)) / len(rating_valid_y)
lamda = 4
iterNum = 0
beta_u = beta_u0
beta_i = beta_i0
alpha = alpha0
while iterNum > -1:
    new_alpha = sum([Rui[(t['reviewerID'], t['itemID'])] - beta_u[t['reviewerID']] - beta_i[t['itemID']] for t in train_data]) / len(train_data)
    new_beta_i, new_beta_u = defaultdict(int), defaultdict(int)
    for u in Iu.keys():
        new_beta_u[u] = sum([ Rui[(u,i)] - new_alpha - beta_i[i] for i in Iu[u]]) / (lamda + len(Iu[u]))
    for i in Ui.keys():
        new_beta_i[i] = sum([ Rui[(u,i)] - alpha - new_beta_u[u] for u in Ui[i]]) / (lamda + len(Ui[i]))
    covariance_alpha = abs(alpha - new_alpha)**2
    covariance_beta_i, covariance_beta_u = 0,0
    for i in beta_i:
        covariance_beta_i += abs(beta_i[i] - new_beta_i[i])**2
    for u in beta_u:
        covariance_beta_u += abs(beta_u[u] - new_beta_u[u])**2
    covariance = numpy.sqrt(covariance_alpha + covariance_beta_i + covariance_beta_u)
    covariance = [(alpha - new_alpha)**2] + [(b-nb)**2 for b,nb in zip(beta_i.values(), new_beta_i.values())] + [(b-nb)**2 for b,nb in zip(beta_u.values(), new_beta_u.values())]
    covariance = sum(numpy.sqrt(covariance))
    print "covariance is "+str(covariance)
    if(alpha==new_alpha and beta_i == new_beta_i and beta_u == new_beta_u):
    #if(covariance < 1e-10):
        break
    else:
        alpha, beta_i, beta_u, iterNum = new_alpha, new_beta_i, new_beta_u, iterNum+1
        print "Finish iter " + str(iterNum) + "with lamda " + str(lamda)
# In[] validation
rating_valid_parameters = []
for d in valid_data:
    rating_valid_parameters.append([alpha, beta_u[d['reviewerID']], beta_i[d['itemID']]])

rating_valid_predict = [sum(para) for para in rating_valid_parameters]
rating_valid_MSE = mse(rating_valid_predict, rating_valid_y)
print "MSE of validation set is "+ str(rating_valid_MSE)
mylib.saveData('1M_train_rating',[alpha, beta_u, beta_i])
