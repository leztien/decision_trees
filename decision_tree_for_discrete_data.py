

"""
Data-Maker of discreete multiclass data for
Decision Tree for discreet multiclass data (BUG)
"""


def make_data(m=1000, n=5, k=3, seed=None):
    import numpy as np
    """make discreet data for a multi-class Decision Tree Classifier"""
    #random seed
    if seed is True:
        seed = __import__("random").randint(0, 1000)
        print("seed =", seed)
    if seed:
        np.random.seed(int(seed))
        
    #make data
    df = 5
    max_values = np.ceil(np.random.chisquare(df=df, size=n)).astype("uint16")
    pp = np.random.uniform(low=0, high=1, size=len(max_values))
    X = np.random.binomial(n=max_values, p=pp, size=(m, len(max_values)))
    weights = np.random.uniform(low=0, high=1, size=n)
    weights **= n+k
    negative_weights = np.random.choice([-1,1], size=n, replace=True)
    weights *= negative_weights
    y = ((X+1)*weights).sum(axis=1)
    bins = np.quantile(y, q=np.linspace(0,1, num=k+1)[1:-1])
    y = np.digitize(y, bins=bins)
    return(X,y)



#######################################################################


import numpy as np


"""METRICS"""
def gini(counts):
    from functools import reduce
    from operator import sub
    total = sum(counts)
    seq = (1,) + tuple((count/total)**2 for count in counts)
    ans = reduce(sub, seq)
    return ans


def Gini(nx, j, value, X, y):
    """Gini avg of an internal node"""
    mask = X[nx,j] == value
    if len(nx)==0:
        from math import inf
        return inf
    y_false = tuple(y[nx][~mask])
    y_true = tuple(y[nx][mask])
    
    assert len(y_false)+len(y_true) == len(mask)
    
    counts_false = [y_false.count(label) for label in set(y_false)]
    counts_true = [y_true.count(label) for label in set(y_true)]
    
    gini_false = gini(counts_false)
    gini_true = gini(counts_true)
    
    proportion_false = len(y_false) / len(mask)
    proportion_true = len(y_true) / len(mask)
    
    ans = gini_false * proportion_false + gini_true * proportion_true
    return ans


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
    def __init__(self, feature, value):
        self.feature = feature
        self.value = value
        self.next_nodes = None
    def forward(self, x):
        if int(x[self.feature]) != self.value:
            ans = self.next_nodes[0].forward(x)
        else:
            ans = self.next_nodes[1].forward(x)
        return ans



class Leaf:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
    def forward(self, x):
        ypred = self.predicted_class
        return ypred



class Tree:
    def __init__(self, metric='gini'):
        self.metric = str(metric).lower(); assert self.metric in ("gini","entropy","gain","information_gain"), "bad metric"
        self.root = None
        self.graph = ["digraph Tree {node [shape=box];",]
        self._nodes_counter = 0
        self._depth_counter = 0

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

    def add_node(self, nx, previous_node_feature=None):
        counts = [(y[nx]==k).sum() for k in self.classes]
        
        """select metric"""
        if self.metric == 'gini':
            g = gini(counts)
            features = set(range(self.n)).difference([previous_node_feature])
            print(len(nx), counts, self._nodes_counter)
            Ginis = list()
            for j in features:
                values = sorted(set(X[:,j]))
                for v in values:
                    G = Gini(nx, j, v, self.X, self.y)
                    Ginis.append((j,v,G))
            G = sorted(Ginis, key=(lambda t:t[-1]))[0]   # best Gini
            base_case_condition = (g <= G[-1]*1.1) or (sum(counts) <= 2) or self._depth_counter > 10
            #  * (1+1E-10)  prevents a bug. IDK wether it is the correct solution
            
        else:
            g = 0
            Gains = list()
            for j in range(self.n):
                G = information_gain(self.y[nx], self.X[nx,j])
                Gains.append((j,G))
            G = sorted(Gains, key=(lambda t:t[-1]))[-1]  # the best IG is at the end
            base_case_condition = G[-1] < 1E-10   # works also if ==0
            
        
        """the recursion section"""
        #base case:
        if base_case_condition:    
            leaf = Leaf(predicted_class=np.array(counts).argmax())
            leaf._node_number = self._nodes_counter
            self._nodes_counter += 1
#            if self.metric == 'gini':  # graphviz
#                txt = 'gini={0:.3f}\n{1:}/{2:}\nclass {3:}'.format(g, *counts, leaf.rule)
#                self.graph.append('{} [color="black", fillcolor="green", style="filled", fontsize=10, label="{}"]'.format(leaf._node_number, txt))
            return leaf
        #non-base case
        else: 
            feature,value = G[0], G[1]
            thisnode = Node(feature=feature, value=value)
            thisnode._node_number = self._nodes_counter
            self._nodes_counter += 1
            next_nodes = [self.add_node(nx[self.X[nx,j] != value], previous_node_feature=thisnode.feature),
                          self.add_node(nx[self.X[nx,j] == value], previous_node_feature=thisnode.feature)]
            thisnode.next_nodes = next_nodes  # note the order - for indexing puropses
            self._depth_counter += 1
            return thisnode
            #graphviz
            if self.metric == 'gini':
                angles = [45, -45]
                branches = [True, False]
                for (node,angle,branch) in zip(next_nodes, angles, branches):
                    self.graph.append('{} -> {} [labeldistance=2.5, labelangle={}, fontsize=8, headlabel="{}"]'.format(thisnode._node_number, node._node_number, angle, branch))
                txt = 'x{} = True\nGini={:.3f}\n{}/{}'.format(thisnode.rule, G[1], *counts)
                self.graph.append('{} [color="black", fillcolor="orange", style="filled", fontsize=10, label="{}"]'.format(thisnode._node_number, txt))
            return thisnode

#############################################################################
    
    
############################################################################################
############################################################################################


X,y = make_data(m=1000, n=3, k=3, seed=True)
        

tree = Tree(metric='gini').fit(X,y)
ypred = tree.predict(X)
acc = np.equal(y, ypred).mean()
print("accuracy =", acc, "\tnumber of nodes =", tree._nodes_counter)






from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

Xtrain,Xtest, ytrain,ytest = train_test_split(X,y)

md = DecisionTreeClassifier(min_samples_split=5, min_samples_leaf=5).fit(Xtrain, ytrain)

acc = md.score(Xtest, ytest)
print("acc", acc)


md.feature_importances_

P = md.predict_proba(X)


