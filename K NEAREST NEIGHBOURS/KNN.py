import numpy as np

#====== CLASS FOR IMPLEMENTING K-Nearest Neighbours ======#

class KNN:

    def set(self,Xin,Yin,kin=5):
        self.X=Xin
        self.n=self.X.shape[1] # Number of features
        self.Y=np.array(Yin,dtype=np.int64)  # To store the ouptuts as integer
        self.noc=np.unique(Yin).size  # number of classes
        self.m=Yin.size # number of training examples
        self.k=kin      # number of nearest neighbours to be checked
 
    def sq_dist(self,X1,X2):
        """
        Here X2 is for a single point
        and X1 is any array of points
        This function returns an array containing
        square of distances of X2 from every example in X1
        """
        return np.sum(np.power((X1-X2),2),axis=1)

    def k_neighbours(self,Xin):
        arr=np.array(self.sq_dist(self.X,Xin))
        ind=np.argsort(arr)
        return self.Y[ind[:self.k]]

    def mode(self,arr):
        return np.argmax(np.bincount(arr))

    def predict(self,Xin):
        # to return the class having maximum frequency
        sz=Xin.shape[0]
        Yout=np.zeros(sz)
        for i in range(sz):
            Yout[i]=self.mode(self.k_neighbours(Xin[i,:]))
        return Yout

    def accuracy(self,test):
        prediction=self.predict(test.X)
        acc=(sum(prediction==test.Y)/test.m)*100
        return acc