def Sigmoid(z):
    return 1/(1+np.exp(-z))

#====== CLASS FOR IMPLEMENTING LOGISTIC REGRESSION ======#

class logReg:
    def __init__(self,Xin,Yin):
        self.Y=Yin
        self.X=np.hstack([np.ones([Xin.shape[0],1]),Xin])  # adding a column of 1's at the beginning of X
        self.theta=np.random.rand(Xin.shape[1]+1)       
    
    def Cost(self,LAMBDA=0):
        h=Sigmoid(np.dot(self.X,self.theta))
        reg=(LAMBDA/(2*self.Y.size))*np.dot(self.theta.T,self.theta)
        J=-(np.dot(self.Y.T,np.log(h))+np.dot(1-self.Y.T,np.log(1-h)))/self.Y.size + reg # Cost after Regularisation
        return J
 
    def predict(self,X):
        return Sigmoid(np.dot(X,self.theta))>=0.5

    def GradientDescent(self,LAMBDA=0,alpha=0.1,iter=2000):
        for i in range(1,iter):
            h=Sigmoid(np.dot(self.X,self.theta))
            P=np.dot(self.X.T,(h-self.Y))
            grad=(alpha/self.Y.size)*P+(LAMBDA/self.Y.size)*self.theta
            grad[0]=(alpha/self.Y.size)*P[0]
            self.theta=self.theta-grad
            #print("Cost in "+str(i)+"th iteration: "+str(self.Cost(LAMBDA)))

    def accuracy(self,test):
        prediction=self.predict(test.X)
        acc=(sum(prediction==test.Y)/test.Y.size)*100
        return acc

    def params(self):
        return self.theta
  
    
