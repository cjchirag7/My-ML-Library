#!/usr/bin/env python
# coding: utf-8

# In[5]:

#====== CLASS FOR IMPLEMENTING LINEAR REGRESSION ======#

class linReg:
    
    def __init__(self,Xin,Yin):
        self.Y=Yin
        self.X=np.hstack([np.ones([Xin.shape[0],1]),Xin])  # adding a column of 1's at the beginning of X
        self.theta=np.random.rand(Xin.shape[1]+1)       
    
    def Cost(self,LAMBDA=0):
        h=np.dot(self.X,self.theta)
        reg=(LAMBDA/(2*self.Y.size))*np.dot(self.theta.T,self.theta)
        J=np.sum(np.power((h-self.Y),2))/(2*self.Y.size)+reg
        return J
 
    def predict(self,X):
        return np.dot(X,self.theta)

    def GradientDescent(self,LAMBDA=0,alpha=0.1,iter=2000):
        for i in range(1,iter):
            h=np.dot(self.X,self.theta)
            P=np.dot(self.X.T,(h-self.Y))
            grad=(alpha/self.Y.size)*P+(LAMBDA/self.Y.size)*self.theta
            grad[0]=(alpha/self.Y.size)*P[0]
            self.theta=self.theta-grad
            #print("Cost in "+str(i)+"th iteration: "+str(self.Cost(LAMBDA)))
    
    def miniGradientDescent(self,batch_sz,shuffle=True,LAMBDA=0,alpha=0.1,iter=2000):
        if(shuffle):
            np.random.shuffle([self.X,self.Y])        
        for i in range(1,iter):
            for j in range(0,Y.size,batch_sz):
                X_mini=self.X[j:j+batch_sz,:]
                Y_mini=self.Y[j:j+batch_sz]
                if(Y_mini.size==0):   
                    break       # to avoid division by zero error
                h=np.dot(X_mini,self.theta)
                P=np.dot(X_mini.T,(h-Y_mini))
                grad=(alpha/Y_mini.size)*P+(LAMBDA/Y_mini.size)*self.theta
                grad[0]=(alpha/Y_mini.size)*P[0]
                self.theta=self.theta-grad
            #print("Cost in "+str(i)+"th iteration: "+str(self.Cost(LAMBDA)))

    def accuracy(self,test):
        prediction=self.predict(test.X)
        Error=((prediction-test.Y)/test.Y)*100 # Error %
        return (100-np.mean(Error))

    def params(self):
        return self.theta
       


# In[ ]:




