#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import LogisticRegression

def normalise(Xin):
        return (Xin-np.mean(Xin,axis=0))/np.std(Xin,axis=0)    # NORMALISING FEATURES

X=np.genfromtxt('LogisticRegressionData1.txt',delimiter=',',usecols=(0,1))
Y=np.genfromtxt('LogisticRegressionData1.txt',delimiter=',',usecols=-1)
sz=X.shape[0]
X=normalise(X)   #normalising X

# separation of training set and test set
test_sz=sz//4  
tr_sz=sz-test_sz   
train=logReg(X[:tr_sz-1,:],Y[:tr_sz-1])
test=logReg(X[tr_sz:,:],Y[tr_sz:])


# Mini-Batch Gradient-Descent
train.miniGradientDescent(10)

# Testing accuracy of prediction on test set
print("The accuracy on the test set is: " + str(train.accuracy(test)) + " %")


# In[ ]:




