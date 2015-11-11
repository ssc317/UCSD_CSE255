# -*- coding: utf-8 -*-
"""
Created on Mon Oct 05 20:34:17 2015

@author: ssc317
"""


# In[1]: save variable and load data file
import pickle
def saveData(fileName, data):
    fileName = fileName + ".pickle"
    print "Save data to the local file " + fileName
    with open(fileName, 'w') as f:
        pickle.dump(data, f)
    print "done"

def loadData(fileName):
    print "Load data from local"
    fileName = fileName + ".pickle"
    with open(fileName) as f:
        data = pickle.load(f)
    print "done"    
    return data

# In[2]: clear memory
def clearAll():
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue
        del globals()[var]
# In[3] load data from json
def parseDataFromF(fname):
  for l in open(fname):
    yield eval(l)
def parseDataFromFile(fname):
    return list(parseDataFromF(fname))
# In[]
import matplotlib.pyplot as plt
import numpy

def plot(x,y):
    plt.plot(x,y)
    plt.show()
