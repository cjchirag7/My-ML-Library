import numpy as np

#====== CLASS FOR IMPLEMENTING NEURAL NETWORKS ======#

class NNClassifier:

    def set(self,Xin,Yin,layer_sizes,e=0.12):
        self.X=Xin
        self.n=self.X.shape[1] # Number of features
        self.out=Yin  # To store the ouptuts 
        self.noc=np.unique(Yin).size  # number of classes
        self.m=Yin.size # number of training examples
        self.Y=np.zeros((self.m,self.noc))
        for i in range(0,self.m):
            self.Y[i,int(Yin[i])]=1
        self.l_size=layer_sizes
        self.nol=len(layer_sizes)  # number of hidden layers
        self.w=np.random.rand(self.nol+1,max(np.max(layer_sizes),self.n),max(np.max(layer_sizes),self.n))*2*e-e 
        # Random initialisation of weights
        self.b=np.random.rand(self.nol+1,max(np.max(layer_sizes),self.n))*2*e-e
        self.grad_w=np.zeros((self.nol+1,max(np.max(layer_sizes),self.n),max(np.max(layer_sizes),self.n)))
        # Initialisation of gradients of weights to 0
        self.grad_b=np.zeros((self.nol+1,max(np.max(layer_sizes),self.n)))

    def Sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def der_Sigmoid(self,z):
        return z*(1-z)

    def get_w(self,L):
        nocol=self.l_size[L-2]  # number of rows in weight matrix
        if L!=(self.nol+1):
            nor=self.l_size[L-1]    
        if L==1:
            nocol=self.n
        elif L==(self.nol+1):
            nor=self.noc
        return self.w[L-1,:nor,:nocol]

    def get_b(self,L):
        if L==(self.nol+1):
            nor=self.noc
        else:
            nor=self.l_size[L-1]
        return self.b[L-1,:nor]

    def predict(self,X):
        # Feed-Forward
        z=np.dot(X,self.get_w(1).T)+self.get_b(1)
        a=self.Sigmoid(z)
        for i in range(1,self.nol):
            z=np.dot(a,self.get_w(i+1).T)+self.get_b(i+1)
            a=self.Sigmoid(z)
        z=np.dot(a,self.get_w(self.nol+1).T)+self.get_b(self.nol+1)
        a=self.Sigmoid(z)
        return np.argmax(a, axis=1)
    
    def Cost(self,LAMBDA=0):
        # Feed-Forward
        z=np.dot(self.X,self.get_w(1).T)+self.get_b(1)
        a=self.Sigmoid(z)
        for i in range(1,self.nol):
            z=np.dot(a,self.get_w(i+1).T)+self.get_b(i+1)
            a=self.Sigmoid(z)
        z=np.dot(a,self.get_w(self.nol+1).T)+self.get_b(self.nol+1)
        a=self.Sigmoid(z)
        # Regularisation Term
        reg=(LAMBDA/(2*self.m))*(np.sum(np.sum(np.power(self.get_w(1),2)))) 
        for i in range(1,self.nol):
            reg+=(LAMBDA/(2*self.m))*(np.sum(np.sum(np.power(self.get_w(i+1),2))))
        reg+=(LAMBDA/(2*self.m))*(np.sum(np.sum(np.power(self.get_w(self.nol+1),2))))
        # Cost after Regularisation
        J=-(np.sum(self.Y*np.log(a))+np.sum((1-self.Y)*np.log(1-a)))/self.m + reg 
        return J

    def calc_grad(self,Xin,Yin,alpha,LAMBDA=0):
        m_in=Yin.shape[0]
        delta=np.zeros((self.nol+1,max(np.max(self.l_size),self.noc)))
        for t in range(0,m_in):
            # Feed-Forwar]
            z=np.random.rand(self.nol+1,max(np.max(self.l_size),self.noc))
            z[0,:self.l_size[0]]=np.dot(Xin[t,:],self.get_w(1).T)+self.get_b(1)
            a=np.random.rand(self.nol+1,max(np.max(self.l_size),self.noc))
            a[0,:self.l_size[0]]=self.Sigmoid(z[0,:self.l_size[0]])
            for i in range(1,self.nol):
                z[i,:self.l_size[i]]=np.dot(a[i-1,:self.l_size[i-1]],self.get_w(i+1).T)+self.get_b(i+1)
                a[i,:self.l_size[i]]=self.Sigmoid(z[i,:self.l_size[i]])
            z[-1,:self.noc]=np.dot(a[self.nol-1,:self.l_size[-1]],self.get_w(self.nol+1).T)+self.get_b(self.nol+1)
            a[-1,:self.noc]=self.Sigmoid(z[-1,:self.noc])
            #Back Propagation
            delta[-1,:self.noc]=(a[-1,:self.noc]-Yin[t,:])*self.der_Sigmoid(a[-1,:self.noc])
            for j in range(self.nol-1,-1,-1):
                if(j==self.nol-1):
                    sz=self.noc
                else:
                    sz=self.l_size[j+1]
                delta[j,:self.l_size[j]]=np.dot(self.get_w(j+2).T,delta[j+1,:sz])*self.der_Sigmoid(a[j,:self.l_size[j]])
            for k in range(0,self.nol):
                nocol=self.l_size[k-1]  # number of rows in weight matrix
                nor=self.l_size[k]    # number of cols in weight matrix
                act=np.array(a[k-1,:self.l_size[k-1]]) [np.newaxis]  # convert 1-D list to 2-D array
                if k==0:
                    nocol=self.n
                    act=np.array(Xin[t,:]) [np.newaxis] 
                elif k==(self.nol-1):
                    nor=self.noc
                D=np.array(delta[k,:nor]) [np.newaxis]
                self.grad_w[k,:nor,:nocol]+=np.dot(D.T,act)
                self.grad_b[k,:nor]+=delta[k,:nor]
        # Dividing gradients by m and applying regularisation
        self.grad_w=((alpha*self.grad_w)+(LAMBDA*self.w))/m_in
        self.grad_b=(alpha*self.grad_b)/m_in

    def GradientDescent(self,LAMBDA=0,alpha=0.03,iter=1000):
        for i in range(1,iter):
            self.calc_grad(self.X,self.Y,alpha,LAMBDA)
            self.w=self.w-self.grad_w
            self.b=self.b-self.grad_b
            #print("Cost in "+str(i)+"th iteration: "+str(self.Cost(LAMBDA)))
            
    def miniGradientDescent(self,batch_sz,shuffle=True,LAMBDA=0,alpha=0.03,iter=500):
        if(shuffle):
            np.random.shuffle([self.X,self.out]) 
            self.Y=np.zeros((self.m,self.noc))
            for i in range(0,self.m):
                self.Y[i,int(self.out[i])]=1       
        for i in range(1,iter):
            for j in range(0,self.m,batch_sz):
                X_mini=self.X[j:j+batch_sz,:]
                Y_mini=self.Y[j:j+batch_sz]
                sz=Y_mini.size
                if(Y_mini.size==0):   
                    break       # to avoid division by zero error
                self.calc_grad(X_mini,Y_mini,alpha,LAMBDA)
                self.w=self.w-self.grad_w
                self.b=self.b-self.grad_b          
            #print("Cost in "+str(i)+"th iteration: "+str(self.Cost(LAMBDA)))
            
    def accuracy(self,test):
        prediction=self.predict(test.X)
        acc=(sum(prediction==test.out)/test.m)*100
        return acc

    def print_params(self):
        for i in range(1,self.nol+2):
            print("Layer "+str(i)+" : ")
            print("Weights: "+str(self.get_w(i)))
            print("Bias: "+str(self.get_b(i)))