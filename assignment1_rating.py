# In[] load 1M data
import mylib


[data_] = mylib.loadData('../1Mtrain')
train_data = data_[:900000]
valid_data = data_[900000:]
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

alpha0 = 0 #numpy.random.rand() #4.21824
beta_u0 = { uid: 0 for uid in Iu.keys()}
beta_i0 = { iid: 0 for iid in Ui.keys()}
rating_valid_y = [d['rating'] for d in valid_data]
def mse(rating_valid_predict, rating_valid_y):
        return sum((a-b)**2 for a,b in zip(rating_valid_predict,rating_valid_y)) / len(rating_valid_y)
lamdas = [1,2,3,4,5,6,7,8,9,10]
thetas = []
MSEs = []
covariance = 0
for lamda in lamdas:
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
        new_covariance = [(alpha - new_alpha)**2] + [(b-nb)**2 for b,nb in zip(beta_i.values(), new_beta_i.values())] + [(b-nb)**2 for b,nb in zip(beta_u.values(), new_beta_u.values())]
        new_covariance = sum(numpy.sqrt(new_covariance))
        print "covariance is "+str(new_covariance)
        #if(alpha==new_alpha and beta_i == new_beta_i and beta_u == new_beta_u):
        #if(covariance < 1e-10):
        if(covariance == new_covariance):
            break
        else:
            alpha, beta_i, beta_u, iterNum, covariance = new_alpha, new_beta_i, new_beta_u, iterNum+1, new_covariance
            print "Finish iter " + str(iterNum) + " with lamda " + str(lamda)
        rating_valid_parameters = []
        for d in valid_data:
            rating_valid_parameters.append([alpha, beta_u[d['reviewerID']], beta_i[d['itemID']]])
        rating_valid_predict = [sum(para) for para in rating_valid_parameters]
        rating_valid_MSE = mse(rating_valid_predict, rating_valid_y)
        MSEs.append(rating_valid_MSE)
        thetas.append([alpha,beta_u,beta_i])
        print "MSE of validation set is "+ str(rating_valid_MSE)
print MSEs
# In[] validation
index = thetas.index(min(MSEs))
[alpha,beta_u,beta_i] = thetas[index]
rating_valid_parameters = []
for d in valid_data:
    rating_valid_parameters.append([alpha, beta_u[d['reviewerID']], beta_i[d['itemID']]])

rating_valid_predict = [sum(para) for para in rating_valid_parameters]
rating_valid_MSE = mse(rating_valid_predict, rating_valid_y)
print "MSE of validation set is "+ str(rating_valid_MSE) + " with lamda "+ str(index+1)
mylib.saveData('1M_train_rating',[alpha, beta_u, beta_i])
