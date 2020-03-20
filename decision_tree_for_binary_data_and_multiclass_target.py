"""
Decision Tree Classifier for binary data and multiclass target
"""

def make_data(m=100, n=5, k=3, make_identical_observations_consistent=False, seed=None):
    import numpy as np
    
    """make binary data for a binary Decision Tree Classifier"""
    #random seed
    if seed is True:
        seed = __import__("random").randint(0, 1000)
        print("seed =", seed)
    if seed:
        np.random.seed(int(seed))

    #make data
    pp = np.random.random(size=n)  # proportions of True in each feature
    X = np.random.binomial(n=1, p=pp, size=(m,n))
    Xoriginal = X.copy().astype('float')  # Xoriginal will be returned; mangle the X

    #reverse the boolean values of some features (e.g. is_old >> is_new)
    def func(X):
        mask = np.random.binomial(n=1, p=0.25, size=n).astype("uint8")
        ff = np.array([np.vectorize(lambda x:x), np.logical_not])[mask]
        Xbool = X.astype("bool")
        Xnew = np.empty_like(X, dtype='bool')
        for i,row in enumerate(Xbool):
            Xnew[i] = [f(x) for f,x in zip(ff,row)]
        return Xnew.astype('uint8')

    X = func(X)

    #f!ck up a certain proportion of each feature by reversing the boolean value in a coresponding cell
    high = (1/n)**2  # maximum allowed proportion to be mangled (the lower n the higher this proportion)
    pp = np.random.uniform(low=0, high=high, size=n)  # proportion of each feature to be mangled
    g = (np.random.permutation(m)[:int(m*p)] for p in pp)
    for j,nx in enumerate(g):
        X[nx,j] = np.logical_not(X[nx,j])  # reverse the boolean feature in certain cells
    
    #weights
    weights = np.random.uniform(0,1, size=n)
    X = X * weights
    
    #compute the target
    y = X.sum(axis=1)
    
    #discretize the target
    q = np.linspace(0,1, num=k+1)[1:-1]
    breaks = np.quantile(y, q=q)
    y = np.digitize(y, bins=breaks).astype('uint8')
    
    if 0 in np.bincount(y):
        from warnings import warn
        warn("One or more classes is missing in the target", Warning)
    
    if make_identical_observations_consistent:
        def make_consistent(X,y):
            """make an unsplittable (degenerate) dataset splittable by assigning a mode label to all identical observations"""
            from scipy.stats import mode
            unique_observations = frozenset(tuple(x) for x in X)
            masks = [[tuple(x) == unique for x in X] for unique in unique_observations] 
            for mask in masks:
                y[mask] = mode(y[mask]).mode[0]
            return(y)
        y = make_consistent(Xoriginal,y)
    #return X,y
    return(Xoriginal,y)

#######################################################################


import numpy as np


"""METRICS"""
def gini(counts):
    from functools import reduce
    from operator import add
    total = sum(counts)
    if total==0: return 1
    return 1 - reduce(add, ((c/total)**2 for c in counts))

def Gini(nx,j, X, y):
    mask = X[nx,j]==1
    y = y[nx]
    g1 = gini(np.bincount(y[mask]))
    g0 = gini(np.bincount(y[~mask]))
    n1 = len(y[mask])
    n0 = len(y[~mask])
    N = n1+n0
    if N == 0: return 1
    G = (n1/N)*g1 + (n0/N)*g0
    return(G)


def entropy(y):
    assert hasattr(y, '__iter__'), "bad argument"
    from math import log2 as log
    y = list(y) if not isinstance(y, list) else y
    unique_values = frozenset(y)
    n = len(y)
    pp = (y.count(v)/n for v in unique_values)
    E = -sum(p * log(p) for p in pp)
    return E

def information_gain(y,x):
    assert all(hasattr(o, '__iter__') for o in (y,x)) and len(y)==len(x), "bad arguments"
    n = len(y)
    E = entropy(y)
    partitions = ([y for y,x in zip(y,x) if x==v] for v in set(x))
    weighted_average_of_entropies_partitioned = sum(entropy(a)*(len(a)/n) for a in partitions)
    I = E - weighted_average_of_entropies_partitioned
    return(I)
    

class Node:
    def __init__(self, rule):
        self.rule = rule
        self.next_nodes = None
    def forward(self, x):
        j = self.rule
        if int(x[j])==1:
            ans = self.next_nodes[1].forward(x)
        elif int(x[j])==0:
            ans = self.next_nodes[0].forward(x)
        else: raise ValueError("bad value")
        return ans


class Leaf:
    def __init__(self, predicted_class):
        self.predicted_class = self.rule = predicted_class
    def forward(self, x):
        ypred = self.predicted_class
        return ypred


class Tree:
    def __init__(self, metric='gini', max_depth=None):
        self.metric = str(metric).lower(); assert self.metric in ("gini","entropy","gain","information_gain"), "bad metric"
        self.root = None
        self.graph = ["digraph Tree {node [shape=box];",]
        self._nodes_counter = 0
        self._level_counter = 0
        self.max_depth = max_depth
        

    def fit(self, X,y):
        self.X, self.y = X,y
        (m,n) = X.shape
        self.n = n
        self.classes = sorted(set(y))
        nx = np.arange(m)
        self.root = self.add_node(nx)
        self.graph = str.join("\n", self.graph + ['}'])
        return self

    def predict(self, X):
        ypred = [self.root.forward(x) for x in X]
        return ypred

    def add_node(self, nx, previous_node_rule=None):
        counts = [(y[nx]==k).sum() for k in self.classes]
        
        if self.metric == 'gini':
            g = gini(counts)
            features = set(range(self.n)).difference([previous_node_rule])
    
            Ginis = list()
            for j in features:
                G = Gini(nx,j, self.X, self.y)
                Ginis.append((j,G))
            G = sorted(Ginis, key=(lambda t:t[-1]))[0]   # best Gini
            base_case_condition = g <= G[-1] * (1+1E-10) #  * (1+1E-10)  prevents a bug. IDK wether it is the correct solution
        
        else:
            g = 0
            Gains = list()
            for j in range(self.n):
                G = information_gain(self.y[nx], self.X[nx,j])
                Gains.append((j,G))
            G = sorted(Gains, key=(lambda t:t[-1]))[-1]  # the best IG is at the end
            base_case_condition = G[-1] < 1E-10   # works also if ==0
        
        
        #depth counter
        if self.max_depth and self._level_counter >= self.max_depth:
            base_case_condition = True
        
        
        """the recursion section"""
        #base case:
        if base_case_condition:    
            leaf = Leaf(predicted_class=np.array(counts).argmax())
            leaf._node_number = self._nodes_counter
            self._nodes_counter += 1
            if self.metric == 'gini':  # graphviz
                txt = 'gini={0:.3f}\n{1:}\nclass {2:}'.format(g, counts, leaf.rule)
                self.graph.append('{} [color="black", fillcolor="green", style="filled", fontsize=10, label="{}"]'.format(leaf._node_number, txt))
            return leaf
        #non-base case
        else: 
            j = G[0]
            thisnode = Node(rule=j)
            thisnode._node_number = self._nodes_counter
            self._nodes_counter += 1
            self._level_counter += 1
            next_nodes = [self.add_node(nx[self.X[nx,j]==0], previous_node_rule=thisnode.rule), self.add_node(nx[self.X[nx,j]==1], previous_node_rule=thisnode.rule)]
            thisnode.next_nodes = next_nodes  # note the order - for indexing puropses
            self._level_counter -= 1
            
            #graphviz
            if self.metric == 'gini':
                angles = [45, -45]
                branches = [False, True]
                for (node,angle,branch) in zip(next_nodes, angles, branches):
                    self.graph.append('{} -> {} [labeldistance=2.5, labelangle={}, fontsize=8, headlabel="{}"]'.format(thisnode._node_number, node._node_number, angle, branch))
                txt = 'x{} = True\nGini={:.3f}\n{}'.format(thisnode.rule, G[1], counts)
                self.graph.append('{} [color="black", fillcolor="orange", style="filled", fontsize=10, label="{}"]'.format(thisnode._node_number, txt))
            return thisnode

#############################################################################


X,y = make_data(m=200, n=10, k=5, make_identical_observations_consistent=False, seed=True)  


tree = Tree(metric='gini', max_depth=6).fit(X,y)
ypred = tree.predict(X)
acc = np.equal(y, ypred).mean()
print("accuracy =", acc, "\tnumber of nodes =", tree._nodes_counter)


#graphviz
string = tree.graph
try: from graphviz import Source
except: print("Unable to import graphviz")
else:
    graph = Source(string, filename="image", format="png")
    graph.view()



from sklearn.tree import DecisionTreeClassifier
md = DecisionTreeClassifier(max_depth=6).fit(X,y)
acc = md.score(X,y)
print("sklearn acc", acc)


