def make_data(m=100, n=3, make_identical_observations_consistent=False, balanced=True, seed=None):
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

    #compute the target
    y = X.sum(axis=1)

    #balanced or imbalanced dataset?
    lower = abs(float(balanced)) if isinstance(balanced,float) else 0.5
    threshold = 0.5 if balanced is True else np.random.uniform(lower, 0.99)
    q = np.quantile(y, q=threshold)
    y = (y >= q).astype('uint8')
    
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

def gini(cc):
    total = sum(cc)
    if total==0: return 1
    return 1 - (cc[0]/total)**2 - (cc[1]/total)**2


def Gini(nx,j):
    xy = tuple(np.c_[X[nx,j]==1, y[nx]].astype('int8').tolist())
    values = ([1,1],[1,0],[0,1],[0,0])
    a,b,c,d = [xy.count(v) for v in values]
    g1 = gini([a,b])
    g0 = gini([c,d])
    n1 = sum([a,b])
    n0 = sum([c,d])
    N = n1+n0
    G = (n1/N)*g1 + (n0/N)*g0
    return(G)


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
    def __init__(self):
        self.root = None
        self.graph = ["digraph Tree {node [shape=box];",]
        self._nodes_counter = 0

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
        cc = [(y[nx]==k).sum() for k in self.classes]
        g = gini(cc)
        features = set(range(self.n)).difference([previous_node_rule])

        Ginis = list()
        for j in features:
            G = Gini(nx,j)
            Ginis.append((j,G))
        G = sorted(Ginis, key=(lambda t:t[-1]))[0]   # best Gini
        
        #base case:
        base_case_condition = g <= G[-1] * (1+1E-10)
        if base_case_condition:    #  * (1+1E-10)  prevents a bug. IDK wether it is the correct solution
            print(g, cc, len(nx))
            leaf = Leaf(predicted_class=np.array(cc).argmax())
            leaf._node_number = self._nodes_counter
            self._nodes_counter += 1
            txt = 'gini={0:.3f}\n{1:}/{2:}\nclass {3:}'.format(g, *cc, leaf.rule)
            self.graph.append('{} [color="black", fillcolor="green", style="filled", fontsize=10, label="{}"]'.format(leaf._node_number, txt))
            return leaf
        else:  #non-base case
            j = G[0]
            thisnode = Node(rule=j)
            thisnode._node_number = self._nodes_counter
            self._nodes_counter += 1
            next_nodes = [self.add_node(nx[self.X[nx,j]==k], previous_node_rule=thisnode.rule) for k in self.classes]
            thisnode.next_nodes = next_nodes  # note the order - for indexing puropses

            #graphviz
            angles = [45, -45]
            branches = [True, False]
            for (node,angle,branch) in zip(next_nodes, angles, branches):
                self.graph.append('{} -> {} [labeldistance=2.5, labelangle={}, fontsize=8, headlabel="{}"]'.format(thisnode._node_number, node._node_number, angle, branch))
            txt = 'x{} = True\nGini={:.3f}\n{}/{}'.format(thisnode.rule, G[1], *cc)
            self.graph.append('{} [color="black", fillcolor="orange", style="filled", fontsize=10, label="{}"]'.format(thisnode._node_number, txt))
            return thisnode

#############################################################################

X,y = make_data(m=300, n=10, make_identical_observations_consistent=True, 
                balanced=True, seed=True)  

tree = Tree().fit(X,y)
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
md = DecisionTreeClassifier().fit(X,y)
acc = md.score(X,y)
print("sklearn acc", acc)
