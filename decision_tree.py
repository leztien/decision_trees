"""
classification tree
handles continuous and categorical variables
"""


from operator import eq, lt, itemgetter
import numpy as np, pandas as pd

from random import randint, choice

import os
from urllib.request import urlopen
from tempfile import mkstemp


def get_module_from_github(url):
    """
    Loads a .py module from github (raw)
    Returns a module object
    """
    with urlopen(url) as response:
        if response.code == 200:
            text = str(response.read(), encoding="utf-8")
    
    _, path = mkstemp(suffix=".py", text=True)
    
    with open(path, mode='wt', encoding='utf-8') as fh:
        fh.write(text)
    
    directory, file_name = os.path.split(path)
    working_dir = os.getcwd()
    os.chdir(directory)
    module = __import__(file_name[:-3])
    os.chdir(working_dir)
    os.remove(path)
    return module


###############################################################


dtypes = dict()
DECIMAL = lt
OBJECT = eq



def is_decimal(x):
    """
    Helper function.
    """
    # if x is an array
    if hasattr(x, '__len__'):
        return any(is_decimal(x) for x in x)
    
    # if x is a single element
    try:
        return not(isinstance(x, str) or
                   x in (None, np.nan, float("nan")) or
                   float(x).is_integer())
    except (TypeError, ValueError, NameError, AttributeError):
        return False



def mode(arr):
    arr = tuple(arr)
    assert len(arr) >= 1, "can't determine the mode from an empty array"
            
    values_counts = [(e, arr.count(e)) for e in set(arr)]
    mx = max(values_counts, key=itemgetter(1))[1]
    classes = [t[0] for t in values_counts if t[1] == mx]
        
    try:
        return min(classes)
    except TypeError:
        return classes[0]



class Node:
    def __init__(self, split=None, prediction=None):
        self.left = None
        self.right = None
        self.split = split  # (j,v)
        self.prediction = prediction
    
    def __call__(self, x):
        if self.split is None:
            return self.prediction
        j, v = self.split
        op = dtypes[j]
        next_node = self.right if op(x[j], v) else self.left
        return next_node(x)
    
    def predict(self, x):
        return self.__call__(x)



def gini(arr):
    arr = tuple(arr)
    if len(arr) == 0: 
        return float('inf')
    return 1.0 - sum((arr.count(e)/len(arr))**2 for e in set(arr))


def get_best_split(X, y, ix):
    # if less than 2 points then no splitting is necessary
    if len(ix) <= 1 or gini(y[ix])==0.0:
        return None
    
    splits = [(gini(y[ix]), None), ] # (g, ((j,v), ix_left, ix_right))
        
    for j in range(X.shape[1]):
        # define the splitting points
        if dtypes[j] is DECIMAL:
            xx = np.sort(X[ix,j])
            splitting_points = (xx[:-1] + xx[1:]) / 2
            op = np.less
        else:
            splitting_points = set(X[ix,j])
            op = np.equal
       
        for v in splitting_points:
            mask = op(X[ix,j], v)
            
            ix_left = ix[~mask]
            ix_right = ix[mask]
            
            gini_left = gini(y[ix_left])
            gini_right = gini(y[ix_right])
            g_weighted = (gini_left * (len(ix_left) / len(ix)) 
                          + gini_right * (len(ix_right) / len(ix)))
            splits.append( (g_weighted, ((j, v), ix_left, ix_right)) )
    
    # get the split with the min gini
    return min(splits, key=itemgetter(0))[1]
        


def make_tree(X, y, ix=None, max_depth=None, **kwargs):
    # determin the data types in each column of X
    global dtypes
    if not dtypes:
        for j in range(X.shape[1]):
            dtypes[j] = DECIMAL if is_decimal(X[:,j]) else OBJECT
    
    # initialize ix (index)
    ix = np.array(range(len(y)), dtype=int) if ix is None else ix
    
    # which depth are you on?
    depth = kwargs.get('depth', 0)
    
    assert len(ix) >= 1, "len(ix) must not be zero" + f" {depth}"
    
    # BASE CASE
    if  len(ix) == 1 or depth == max_depth:
        return Node(prediction=mode(y[ix]))
    
    # get the best split
    best_split = get_best_split(X, y, ix)
    
    if best_split is None:
        return Node(prediction=mode(y[ix]))
    
    # RECURSIVE CASE
    split, ix_left, ix_right = best_split
    node = Node(split=split)
    node.left = make_tree(X, y, ix_left, max_depth, depth=depth+1)
    node.right = make_tree(X, y, ix_right, max_depth, depth=depth+1)
    return node


###TEST#####################################

url = r"https://raw.githubusercontent.com/leztien/toy_datasets/master/make_decision_tree_data.py"
module = get_module_from_github(url)

m = choice([3,10,100,500,1000])
n = randint(1,10)
k = min(randint(2,10), m)
max_depth = randint(3, int(np.log(m*n*k)))

X,y = module.make_decision_tree_data(m, n, k)

print(f"m={m}\tn={n}\tk={len(set(y))}")
print("classes distr:", np.bincount(y))
print("max tree depth:", max_depth, "\n")


from sklearn.tree import DecisionTreeClassifier
md = DecisionTreeClassifier(max_depth=max_depth)
md.fit(X,y)
acc = md.score(X,y)
print("sklearn accuracy:", acc)


tree = make_tree(X,y, max_depth=max_depth)
y_pred = tree.predict(X[0])
y_pred = [tree.predict(x) for x in X]
acc = (y==y_pred).sum() / len(y)
print("my accuracy:", acc)

