
"""
Decision Tree Regressor for continuous data (uses gini)
"""

import numpy as np

def make_data(m=500, n=5, seed=None):
    if seed is True:      #random seed
        seed = __import__("random").randint(0, 1000)
        print("seed =", seed)
    if seed:
        np.random.seed(int(seed))
    X = np.random.normal(loc=0, scale=1, size=(m,n))
    sigmas = np.random.uniform(1,10, size=n)
    X *= sigmas
    X -= X.min(0)
    clips = np.quantile(X, q=np.linspace(0,0.05, num=n), axis=0)
    X = np.clip(X, clips.diagonal(), X.max(0))
    weights = np.random.uniform(-1, 1, size=n)
    y = (X * weights).sum(1)
    y -= y.min()
    return(X.round(2), y.round(1))

#####################################################################


"""METRICS"""
def variance(y):
    if len(y)==0: return np.inf
    var = np.var(y)
    return var

def Variance(nx,j,v, X, y):
    mask = X[nx,j] < v
    y = y[nx]
    v1 = variance(y[mask])
    v0 = variance(y[~mask])
    n1 = len(y[mask])
    n0 = len(y[~mask])
    N = n1+n0
    if n1==0 or n0==0: return np.inf
    V = (n1/N)*v1 + (n0/N)*v0
    return(V)



class Node:
    def __init__(self, j, v):
        self.j = j
        self.v = v
        self.next_nodes = None
    def forward(self, x):
        j = self.j
        v = self.v
        if int(x[j]) < v:
            ans = self.next_nodes[1].forward(x)   #True
        else:
            ans = self.next_nodes[0].forward(x)   #False
        return ans


class Leaf:
    def __init__(self, predicted_value):
        self.predicted_value = predicted_value
    def forward(self, x):
        ypred = self.predicted_value
        return ypred


class Tree:
    def __init__(self, max_depth=None):
        self.root = None
        self._nodes_counter = 0
        self._level_counter = 0
        self._max_level_counter = 0
        self.max_depth = max_depth

        
    def fit(self, X,y):
        self.X, self.y = X,y
        (m,n) = X.shape
        self.n = n
        nx = np.arange(m)
        self.root = self.add_node(nx)
        return self

    def predict(self, X):
        ypred = [self.root.forward(x) for x in X]
        return ypred

    def add_node(self, nx, previous_node_j=None):
        var = variance(self.y[nx])
        
        features = set(range(self.n)).difference([previous_node_j])
        
        #get all the Var's
        Vars = list()
        for j in features:
            values = np.sort(self.X[nx,j])[::len(self.y)//100 or 2]
            values = (values[:-1] + values[1:]) / 2
            if not len(values): continue
            for v in values:
                V = Variance(nx,j,v, self.X, self.y)
                Vars.append((j,v,V))


        if len(Vars)==0:
            base_case_condition = True
        else:
            V = sorted(Vars, key=(lambda t:t[-1]))[0]   # best Gini
            base_case_condition = var <= V[-1] * (1+1E-10) #  * (1+1E-10)  prevents a bug. IDK wether it is the correct solution
    

        #depth counter
        if self.max_depth and self._level_counter >= self.max_depth:
            base_case_condition = True
            
        if len(nx) < 3:
            base_case_condition = True
        
        """the recursion section"""
        #base case:
        if base_case_condition:    
            leaf = Leaf(predicted_value = self.y[nx].mean())
            leaf._node_number = self._nodes_counter
            self._nodes_counter += 1
            return leaf
        #non-base case
        j = V[0]
        v = V[1]
        thisnode = Node(j=j, v=v)
        thisnode._node_number = self._nodes_counter
        self._nodes_counter += 1
        self._level_counter += 1
        self._max_level_counter = max(self._level_counter, self._max_level_counter)
        next_nodes = [self.add_node(nx[self.X[nx,j] >= v], previous_node_j=thisnode.j), 
                      self.add_node(nx[self.X[nx,j] < v], previous_node_j=thisnode.j)]
        thisnode.next_nodes = next_nodes  # note the order - for indexing puropses
        self._level_counter -= 1
        return thisnode

#############################################################################


X,y = make_data(m=1000, n=5, seed=771)  #771  


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y)

md = DecisionTreeRegressor(max_depth=10)
md.fit(Xtrain, ytrain)

rsq = md.score(Xtest, ytest)
print("sklearn RSquared =", rsq.round(2))


#####################################################################

tree = Tree(max_depth=10)
tree.fit(Xtrain, ytrain)
ypred = tree.predict(Xtest)
rsq = r2_score(ytest, ypred).round(2)
print("my RSquared =", rsq, "\tnumber of nodes =", tree._nodes_counter, "\ttree depth =", tree._max_level_counter)
