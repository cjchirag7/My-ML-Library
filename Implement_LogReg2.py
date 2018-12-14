#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import LogisticRegression

def normalise(Xin):
        return (Xin-np.mean(Xin,axis=0))/np.std(Xin,axis=0)    # NORMALISING FEATURES

X=np.genfromtxt('LogisticRegressionData2.txt',delimiter=',',usecols=(0,1,2))
np.random.shuffle(X); # Since data given is sorted
Y=X[:,-1].copy()
X=X[:,:-1].copy()
sz=X.shape[0]
X=normalise(X)   #normalising X
X=np.hstack([np.ones([sz,1]),X])  # adding a column of 1's at the beginning of X

# separation of training set and test set
test_sz=sz//4  
tr_sz=sz-test_sz   
train=logReg(X[:tr_sz-1,:],Y[:tr_sz-1])
Xtest=X[tr_sz:,:]
Ytest=Y[tr_sz:]


# Gradient-Descent
train.GradientDescent(LAMBDA=200)

# Testing accuracy of prediction on test set
prediction=train.predict(Xtest)
acc=(sum(prediction==Ytest)/test_sz)*100
print("The accuracy on the test set is: " + str(acc) + " %")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




