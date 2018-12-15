#====== CLASS FOR IMPLEMENTING LINEAR REGRESSION ======#

class linReg:
    
    def __init__(self,Xin,Yin):
        self.X=Xin
        self.Y=Yin
        self.m=Yin.size # number of training examples
        self.slope=np.random.rand(Xin.shape[1])
        self.inter=np.random.rand(1)
    
    def predict(self,X):
        return np.dot(X,self.slope)+self.inter
    
    def Cost(self,LAMBDA=0):
        h=np.dot(self.X,self.slope)+self.inter
        reg=(LAMBDA/(2*self.m))*(np.sum(np.power(self.slope,2)))
        J=np.sum(np.power((h-self.Y),2))/(2*self.m)+reg
        return J 

    def GradientDescent(self,LAMBDA=0,alpha=0.1,iter=2000):
        for i in range(1,iter):
            h=np.dot(self.X,self.slope)+self.inter
            P=np.dot(self.X.T,(h-self.Y))
            grad_slope=(alpha/self.m)*P+(LAMBDA/self.m)*self.slope
            grad_inter=(alpha/self.m)*np.sum(h-self.Y)
            self.slope=self.slope-grad_slope
            self.inter=self.inter-grad_inter
            #print("Cost in "+str(i)+"th iteration: "+str(self.Cost(LAMBDA)))
    
    def miniGradientDescent(self,batch_sz,shuffle=True,LAMBDA=0,alpha=0.1,iter=2000):
        if(shuffle):
            np.random.shuffle([self.X,self.Y])        
        for i in range(1,iter):
            for j in range(0,Y.size,batch_sz):
                X_mini=self.X[j:j+batch_sz,:]
                Y_mini=self.Y[j:j+batch_sz]
                sz=Y_mini.size
                if(Y_mini.size==0):   
                    break       # to avoid division by zero error
                h=np.dot(X_mini,self.slope)+self.inter
                P=np.dot(X_mini.T,(h-Y_mini))
                grad_slope=(alpha/sz)*P+(LAMBDA/sz)*self.slope
                grad_inter=(alpha/sz)*np.sum(h-Y_mini)
                self.slope=self.slope-grad_slope
                self.inter=self.inter-grad_inter            
            #print("Cost in "+str(i)+"th iteration: "+str(self.Cost(LAMBDA)))

    def accuracy(self,test):
        prediction=self.predict(test.X)
        Error=((prediction-test.Y)/test.Y)*100 # Error %
        return (100-np.mean(Error))

    def params(self):
        return [self.inter,self.slope]




