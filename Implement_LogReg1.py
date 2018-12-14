#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import LogisticRegression



X=np.genfromtxt('LogisticRegressionData1.txt',delimiter=',',usecols=(0,1))
Y=np.genfromtxt('LogisticRegressionData1.txt',delimiter=',',usecols=-1)
sz=X.shape[0]
X=normalise(X)   #normalising X

# separation of training set and test set
test_sz=sz//4  
tr_sz=sz-test_sz   
train=logReg(X[:tr_sz-1,:],Y[:tr_sz-1])
Xtest=X[tr_sz:,:]
Ytest=Y[tr_sz:]


# Gradient-Descent
train.GradientDescent()

# Testing accuracy of prediction on test set
prediction=train.predict(Xtest)
acc=(sum(prediction==Ytest)/test_sz)*100
print("The accuracy on the test set is: " + str(acc) + " %")


# In[ ]:




