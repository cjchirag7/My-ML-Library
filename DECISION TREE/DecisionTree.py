import numpy as np

#====== CLASS FOR IMPLEMENTING DECISION TREE CLASSIFICATION======#

class decTree:

    def set(self,Xin,Yin):
        self.X=Xin
        self.Y=Yin
        self.m=Yin.size # number of training examples
        self.n=Xin.shape[1]
        self.Node=self.build_tree(self.X,self.Y)
        
    def class_counts(self,y):
        """
        This function returns a dictionary having
        the counts of all classes in 'y'
        """
        count_dict={}  # Dictionary 
        for label in y:
            if label not in count_dict:
                count_dict[label]=0
            count_dict[label]+=1
        return count_dict           
    
    def gini_cost(self,X,y):
        """
        This function returns the gini cost in the given branch of tree
        using the formula:
        1 - sum of square of probabilities for all classes
        """
        count_dict = self.class_counts(y)
        gini=1
        for label in count_dict:
            prob = count_dict[label] / float(len(X))
            gini-=(prob*prob)
        return gini

    def gini_split(self, right_X, right_Y, left_X, left_Y, node_cost):
        """
        This function returns the gini of a split as:
        GINI (split,node) = GINI (node) – {PROPORTION (left) * GINI (left)} – {PROPORTION (right) * GINI (right)}
        """
        p = float(len(right_X)) / (len(right_X) + len(left_X))
        return node_cost - p * self.gini_cost(right_X, right_Y) - (1 - p) * self.gini_cost(left_X, left_Y)

    class threshold:
        """
        This class stores an index of a feature 
        and its threshold value  
        It helps in splitting the dataset
        as per the given threshold by telling
        whether it satisfies the condition or not
        """
        def __init__(self,ind,value):
            self.index=ind   
            self.value=value
            
        def match(self,example):
            """
            Returns True when passed example 
            satisfies the threshold condition
            And False when it doesn't
            """
            val=example[self.index]
            if isinstance(val,int) or isinstance(val,float):
                return val>=self.value    # For continuous data
            else:
                return val==self.value    # For labelled data
            
    def split(self,X,y,threshold):
        """
        This function partitions the data into two branchs according to
        the given threshold and RETURNS THE BRANCHES
        (Indirectly, it creates a branch of a BINARY TREE)
        """
        right_X,right_Y,left_X,left_Y=[],[],[],[]
        for i in range(len(X)):   # Traversal through len(X)
            if threshold.match(X[i]):
                right_X.append(X[i])
                right_Y.append(y[i])
            else:
                left_X.append(X[i])
                left_Y.append(y[i])
        return right_X,right_Y,left_X,left_Y
    
    def get_best_split(self,X,y):
        """
        This function finds the best threshold to be taken 
        to have MAXIMUM GINI OF SPLIT 
        """
        best_gain = 0  
        best_threshold = None  
        node_cost = self.gini_cost(X,y)  
        for col in range(self.n):  
            values = set([row[col] for row in X])  
            for val in values:   
                threshold = self.threshold(col, val)
                right_X, right_Y, left_X, left_Y = self.split(X, y, threshold)
                if (len(right_X) == 0 or len(left_X) == 0):
                    continue # No split possible 
                gain = self.gini_split(right_X, right_Y, left_X, left_Y, node_cost)
                if gain >= best_gain:
                    best_gain, best_threshold = gain, threshold
        return best_gain, best_threshold
    
    #====CLASS FOR ANY MIDDLE NODE====#

    class Decision_Node:

        def __init__(self,threshold,right_branch,left_branch):
            self.right_branch=right_branch
            self.left_branch=left_branch
            self.threshold=threshold
    
    def build_tree(self, X, Y):
        """
        This functions do the branching recursively and returns the respective nodes
        """
        gain, threshold=self.get_best_split(X,Y)
        if gain == 0:     # Leaf Node Reached
            return self.class_counts(Y)   # The predictions
        right_X,right_Y,left_X,left_Y=self.split(X,Y,threshold)
        right_branch=self.build_tree(right_X,right_Y)
        left_branch=self.build_tree(left_X,left_Y)
        return self.Decision_Node(threshold,right_branch,left_branch)
            
    def classify(self,Node,example):
        """
        This function is used to classify an example by using the trained decision tree
        """
        if isinstance(Node,dict):  # Leaf Node
            return Node    
        else:
            if(Node.threshold.match(example)):
                return self.classify(Node.right_branch,example)
            else:
                return self.classify(Node.left_branch,example) 
            
    def predict(self,Xtest):    
        predictions=[]
        for example in Xtest:
            d=self.classify(self.Node,example)
            val=list(d.values())
            key=list(d.keys())
            predictions.append(key[val.index(max(val))])
        return np.array(predictions)
    
    def accuracy(self,Xtest,Ytest):        
        predictions=self.predict(Xtest)
        a=np.array(predictions==Ytest)
        acc=np.mean(a)*100
        return acc                 