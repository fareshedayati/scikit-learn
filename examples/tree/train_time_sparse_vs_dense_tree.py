import numpy as np
import time
import sys
sys.path.insert(0, '../../')

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_true
from scipy.sparse import csc_matrix
from sklearn.tree import DecisionTreeClassifier

nr = 10000
nf = 28000

n_test = 100
X_ = np.random.randint(100, size=(nr, nf))
y_ = np.random.randint(2, size=nr)
X_test = np.random.randint(100, size=(n_test, nf))

for j in xrange(X_.shape[1]):
    for i in xrange(X_.shape[0]):
        if np.random.uniform() < 0.95:
            X_[i, j] = 0

start = time.clock()
d = DecisionTreeClassifier(random_state=0, max_depth=3).fit(X_, y_)
end = time.clock()
print "Time to train dense tree  = ", (end - start), "seconds"

start = time.clock()
s = DecisionTreeClassifier(random_state=0, max_depth=3).fit(csc_matrix(X_), y_)
end = time.clock()
print "Time to train sparse tree  = ", (end - start), "seconds"

if False in (d.predict(X_test) == s.predict(X_test)):
    print "They are not predicting the same"

d = d.tree_
s = s.tree_
assert_true((False not in (d.feature == s.feature)) and
                (False not in (d.threshold == s.threshold))and
                (False not in (d.n_node_samples == s.n_node_samples)) and
                (False not in (d.impurity == s.impurity)) and
                (False not in (d.value == s.value)) and
                (False not in (d.children_right == s.children_right)) and
                (False not in (d.children_left == s.children_left))
                , "Sparse different from dense")
assert_array_equal(d.value, s.value, "Sparse different from dense")
