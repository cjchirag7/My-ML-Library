#!/usr/bin/env python
# coding: utf-8

# In[6]:



import numpy as np

def Sigmoid(z):
    return 1/(1+np.exp(-z))

def Cost(X,Y,theta):
    h=Sigmoid(np.dot(X,theta))
    J=-(np.dot(Y.T,np.log(h))+np.dot(1-Y.T,np.log(1-h)))/Y.size
    return J
 
def predict(X,theta):
    p=Sigmoid(np.dot(X,theta))
    return p>=0.5

def GradientDescent(X,Y,theta,alpha=0.1,iter=2000):
    for i in range(1,iter):
        h=Sigmoid(np.dot(X,theta))
        theta=theta-(alpha/Y.size)*np.dot(X.T,(h-Y))
        #print("Cost in "+str(i)+"th iteration: "+str(Cost(Xtr,Ytr,theta)))
    return theta

X=np.genfromtxt('LogisticRegressionData1.txt',delimiter=',',usecols=(0,1))
Y=np.genfromtxt('LogisticRegressionData1.txt',delimiter=',',usecols=-1)
sz=X.shape[0]
X=(X-np.mean(X,axis=0))/np.std(X,axis=0)       # NORMALISING FEATURES
X=np.hstack([np.ones([sz,1]),X])  # adding a column of 1's at the beginning of X

# separation of training set and test set
test_sz=sz//4  
tr_sz=sz-test_sz   
Xtr=X[:tr_sz-1,:]
Ytr=Y[:tr_sz-1]
Xtest=X[tr_sz:,:]
Ytest=Y[tr_sz:]

theta=np.zeros(3)   # initializing parameters

#print(Cost(Xtr,Ytr,theta))

# Gradient-Descent
theta=GradientDescent(X,Y,theta)

# Testing accuracy of prediction on test set
prediction=predict(Xtest,theta)
acc=(sum(prediction==Ytest)/test_sz)*100
print("The accuracy on the test set is: " + str(acc) + " %")


# In[ ]: