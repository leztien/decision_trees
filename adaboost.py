
import numpy as np



def load_from_github(url):
    from urllib.request import urlopen
    from os import remove
    
    obj = urlopen(url)
    assert obj.getcode()==200,"unable to open"

    s = str(obj.read(), encoding="utf-8")
    NAME = "_temp.py"
    with open(NAME, mode='wt', encoding='utf-8') as fh: fh.write(s)
    module = __import__(NAME[:-3])
    remove(NAME)
    return module



def make_data(m, n):
    """wrapper function"""
    path = r"https://raw.githubusercontent.com/leztien/synthetic_datasets/master/make_data_for_decision_trees.py"
    module = load_from_github(path)

    X,y = module.make_data_for_decision_trees(m, n, k=2, proportion_of_binary_features = 1.0)
    y = np.array(y)
    
    m,n = X.shape
    
    features = set(range(n))
    
    for j in tuple(features):
        p = max(sum(X[:,j] == y) / m, sum(X[:,j] != y) / m) 
        if p >= 0.9:
            features.remove(j)
    
    # Remove features
    X = X[:, list(features)]
    
    # Add random features
    new = np.random.randint(0,2, size=(m, n - len(features)), dtype=X.dtype)
    X = np.hstack([X, new])
    return X,y




def get_best_feature(X, y, feature_subset=None):
    m,n = X.shape
    subset = set(feature_subset or range(n))
    ginis = dict()
    
    for j in subset:
        mask = X[:,j] == 1
        n1 = sum(mask)
        n0 = m - n1
        
        p1 = sum(y[mask]==1) / n1
        p0 = sum(y[~mask]==1) / n0

        g1 = 1 - p1**2 - (1-p1)**2
        g0 = 1 - p0**2 - (1-p0)**2
        
        G = (n1/m)*g1 + (n0/m)*g0
        ginis[j] = G

    mn = min(ginis.values())
    return [k for k,v in ginis.items() if v == mn][0]




class Stump:
    def __init__(self, j, X, y, nx=None):
        self.feature = j
        nx = ... if nx is None else nx
        self.X, self.y = X[nx], y[nx]
        self._compute()
    
    def _compute(self):
        mask = self.X[:, self.feature] == 1
        ypred = np.bincount(self.y[mask]).argmax()
        self.prediction = {1: int(ypred), 0: int(not ypred)}
        # get predictions
        ypreds = [self.predict(x) for x in self.X]
        self.error = sum(y != ypreds) / len(self.X)
        self.say = (1/2) * np.log((1 - self.error) / self.error)
        mask = ypreds == self.y
        multipliers = np.array([-1,] * len(self.X))
        multipliers[mask] = 1
        self.sample_weights = (1/len(self.X)) * np.exp(self.say * multipliers)
        self.sample_weights = self.sample_weights / self.sample_weights.sum()
        
    def predict(self, x):
        return self.prediction[x[self.feature]]

    def sample_new_observations(self):
        m,n = self.X.shape
        nx = np.searchsorted(self.sample_weights.cumsum(), np.random.rand(m))
        assert max(nx) < m
        return nx


class AdaBoost:
    def __init__(self, stumps):
        self.stumps = stumps
        self.cumsay = sum(stump.say for stump in self.stumps)
    
    def predict(self, x):
        S = 0.0
        for stump in self.stumps:
            ypred = stump.predict(x)
            if ypred == 1:
                S += stump.say
        return S / self.cumsay
    
    def accuracy(self, X, y):
        ypreds = [int(self.predict(x) >= 0.5) for x in X]
        return sum(y == ypreds) / len(y)


############################################################

X,y = make_data(100, 10)
m,n = X.shape


# TEST
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y)

for c in (DecisionTreeClassifier, RandomForestClassifier):
    md = c().fit(Xtrain, ytrain)
    
    acc = md.score(Xtest, ytest)
    print(f"Acc of {c}:", acc)


######################################################

# Algorithm

features = set(range(n))
nx = list(range(m))
stumps = []

while features:
    best_feature = get_best_feature(X, y, features)
    features.remove(best_feature)
    
    stump = Stump(best_feature, X, y, nx)
    stumps.append(stump)
    
    #sample new observations
    nx = stump.sample_new_observations()

   
adaboost = AdaBoost(stumps)  
acc = adaboost.accuracy(X,y)
print(acc)
