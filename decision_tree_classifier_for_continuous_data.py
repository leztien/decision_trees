
"""
Decision Tree for continuous data and categorical target
uses gini
"""


import numpy as np

def make_data(m=500, n=5, k=3, seed=None):
    #random seed
    if seed is True:
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
    y = (y - y.mean()) / y.std(ddof=0)
    errors = np.random.normal(loc=0, scale=0.0001, size=m)
    y += errors
    breaks1 = np.quantile(y, q=np.linspace(0,1,num=k+1))[1:-1]
    breaks2 = np.arange(y.min(), y.max()+0.1, k)[1:]
    breaks = sorted((breaks1 + breaks2) / 2)
    y = np.digitize(y, bins=breaks).astype('uint8')
    return(X.round(2), y)

#####################################################################



"""METRICS"""
def gini(counts):
    from functools import reduce
    from operator import add
    total = sum(counts)
    if total==0: return 1
    return 1 - reduce(add, ((c/total)**2 for c in counts))

def Gini(nx,j,v, X, y):
    mask = X[nx,j] < v
    y = y[nx]
    g1 = gini(np.bincount(y[mask]))
    g0 = gini(np.bincount(y[~mask]))
    n1 = len(y[mask])
    n0 = len(y[~mask])
    N = n1+n0
    if N == 0: return 1
    G = (n1/N)*g1 + (n0/N)*g0
    return(G)


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
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
    def forward(self, x):
        ypred = self.predicted_class
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
        self.classes = sorted(set(y))
        nx = np.arange(m)
        self.root = self.add_node(nx)
        return self

    def predict(self, X):
        ypred = [self.root.forward(x) for x in X]
        return ypred

    def add_node(self, nx, previous_node_j=None):
        counts = [(self.y[nx]==k).sum() for k in self.classes]
        
        g = gini(counts)
        features = set(range(self.n)).difference([previous_node_j])
        
        #get all the Gini's
        Ginis = list()
        for j in features:
            values = np.sort(self.X[nx,j])[::2]
            values = (values[:-1] + values[1:]) / 2
            if not len(values): continue
            for v in values:
                G = Gini(nx,j,v, self.X, self.y)
                Ginis.append((j,v,G))


        if len(Ginis)==0:
            base_case_condition = True
        else:
            G = sorted(Ginis, key=(lambda t:t[-1]))[0]   # best Gini
            base_case_condition = g <= G[-1] * (1+1E-10) #  * (1+1E-10)  prevents a bug. IDK wether it is the correct solution
    

        #depth counter
        if self.max_depth and self._level_counter >= self.max_depth:
            base_case_condition = True

        
        """the recursion section"""
        #base case:
        if base_case_condition:    
            leaf = Leaf(predicted_class=np.array(counts).argmax())
            leaf._node_number = self._nodes_counter
            self._nodes_counter += 1
            return leaf
        #non-base case
        j = G[0]
        v = G[1]
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


X,y = make_data(m=1000, n=5, k=3, seed=188)  #188


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y)

md = DecisionTreeClassifier(max_depth=10)
md.fit(Xtrain, ytrain)

acc = md.score(Xtest, ytest)
print("sklearn accuracy =", acc)


#####################################################################


tree = Tree(max_depth=10).fit(Xtrain, ytrain)
ypred = tree.predict(Xtest)
acc = np.equal(ytest, ypred).mean()
print("my accuracy =", acc, "\tnumber of nodes =", tree._nodes_counter, "\ttree depth =", tree._max_level_counter)
