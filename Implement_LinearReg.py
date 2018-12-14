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
test_sz=sz//4   # 11
tr_sz=sz-test_sz   # 36
train=linReg(X[:tr_sz-1,:],Y[:tr_sz-1])
Xtest=X[tr_sz:,:]
Ytest=Y[tr_sz:]

# Gradient-Descent
train.GradientDescent()

# Testing accuracy of prediction on test set
prediction=train.predict(Xtest)
print(train.X)
Error=((prediction-Ytest)/Ytest)*100
print("The average error in test set is: " + str(np.mean(Error)) + " %")


# In[ ]




# In[ ]:




