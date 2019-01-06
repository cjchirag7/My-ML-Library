import numpy as np

#====== CLASS FOR IMPLEMENTING K-Means Clustering ======#

class K_clust:

    def set(self,Xin,noc):
        self.X=Xin
        self.m=self.X.shape[0] # number of training examples
        self.k=noc      # number of clusters user requires
        ind=np.random.choice(self.m,self.k) # selects k random indices from our dataset
        self.centroids=self.X[ind,:] # initialise centroids to the values at those indices

    def set_k(self,noc):
        self.k=noc;   # number of clusters       

    def sq_dist(self,X1,X2,ax=1):
        """
        Here X2 is for a single point
        and X1 is any array of points
        This function returns an array containing
        square of distances of X2 from every example in X1
        """
        return np.linalg.norm(X1-X2,axis=ax)

    def assign_clust(self):
        """ 
        This function will assign cluster to each training example, such that it gets assigned to that
        cluster whose centroid is nearest to it
        """
        clust=np.zeros(self.m,dtype=np.int64)
        for i in range(self.m):
            clust[i]=np.argmin(self.sq_dist(self.X[i],self.centroids))
        return clust

    def move_cent(self,clust):
        """ This function will move the centroids to the mean positions of their respective clusters"""
        for i in range(self.k):
            ind_match=np.argwhere(i==(clust)) # To store indices that match with a particular cluster
            self.centroids[i,:]=np.mean(self.X[ind_match,:],axis=0)

    def fit(self,noi=100):
        # to perform K-means clustering and predict for each training example
        for i in range(noi):
            clust=self.assign_clust()
            self.move_cent(clust)
        return clust

    def predict(self,Xtest):
        # To predict clusters of test data as per our trained dataset
        m=Xtest.shape[0]
        clust=np.zeros(m,dtype=np.int64)
        for i in range(m):
            clust[i]=np.argmin(self.sq_dist(Xtest[i],self.centroids))
        return clust