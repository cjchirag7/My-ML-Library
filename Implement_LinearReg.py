#!/usr/bin/env python
# coding: utf-8

# In[3]:

import numpy as np
import LinearRegression

def normalise(Xin):
        return (Xin-np.mean(Xin,axis=0))/np.std(Xin,axis=0)    # NORMALISING FEATURES

X=np.genfromtxt('LinearRegressionData.txt',delimiter=',',usecols=(0,1))
Y=np.genfromtxt('LinearRegressionData.txt',delimiter=',',usecols=-1)
sz=X.shape[0]
X=normalise(X)

# separation of training set and test set
test_sz=sz//4   
tr_sz=sz-test_sz   
train=linReg(X[:tr_sz-1,:],Y[:tr_sz-1])
test=linReg(X[tr_sz:,:],Y[tr_sz:])

# Mini-Batch Gradient-Descent
train.miniGradientDescent(5)

# Testing accuracy of prediction on test set
print("The accuracy on test set is: " + str(train.accuracy(test)) + " %")






