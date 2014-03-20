# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Fares Hedayati <fareshedayat@yahoo.com>
# Licence: BSD 3 clause

# See _tree.pyx for details.

import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

# =============================================================================
# Criterion
# =============================================================================

cdef class Criterion:
    # Internal structures
    cdef DOUBLE_t* y                     # Values of y
    cdef SIZE_t y_stride                 # Stride in y (since n_outputs >= 1)
    cdef DOUBLE_t* sample_weight         # Sample weights

    cdef SIZE_t* samples                 # Sample indices in X, y
    cdef SIZE_t start                    # samples[start:pos] are the samples in the left node
    cdef SIZE_t pos                      # samples[pos:end] are the samples in the right node
    cdef SIZE_t end

    cdef SIZE_t n_outputs                # Number of outputs
    cdef SIZE_t n_node_samples           # Number of samples in the node (end-start)
    cdef double weighted_n_node_samples  # Weighted number of samples
    cdef double weighted_n_left          # Weighted number of samples in the left node
    cdef double weighted_n_right         # Weighted number of samples in the right node

    # The criterion object is maintained such that left and right collected
    # statistics correspond to samples[start:pos] and samples[pos:end].

    # Methods
    cdef void init(self, DOUBLE_t* y,
                         SIZE_t y_stride,
                         DOUBLE_t* sample_weight,
                         SIZE_t* samples,
                         SIZE_t start,
                         SIZE_t end) nogil
    cdef void reset(self) nogil
    cdef void update(self, SIZE_t new_pos) nogil
    cdef double node_impurity(self) nogil
    cdef void children_impurity(self, double* impurity_left, double* impurity_right) nogil
    cdef void node_value(self, double* dest) nogil
    cdef double impurity_improvement(self, double impurity) nogil

# =============================================================================
# Splitter
# =============================================================================

cdef class Splitter:
    # Internal structures
    cdef public Criterion criterion      # Impurity criterion
    cdef public SIZE_t max_features      # Number of features to test
    cdef public SIZE_t min_samples_leaf  # Min samples in a leaf

    cdef object random_state             # Random state
    cdef UINT32_t rand_r_state           # sklearn_rand_r random number state

    cdef SIZE_t* samples                 # Sample indices in X, y
    cdef SIZE_t n_samples                # X.shape[0]
    cdef SIZE_t* features                # Feature indices in X
    cdef SIZE_t* constant_features       # Constant features indices
    cdef SIZE_t n_features               # X.shape[1]
    cdef DTYPE_t* feature_values         # temp. array holding feature values
    cdef DTYPE_t* current_col            # temp. array holding feature values
    cdef UINT32_t* index_to_color
    cdef SIZE_t* tmp_indices              # temp. array of indices
    cdef SIZE_t* hyper_indices              # temp. array of indices

    cdef UINT32_t current_color
    cdef SIZE_t start                    # Start position for the current node
    cdef SIZE_t end                      # End position for the current node

    cdef DTYPE_t* X
    cdef SIZE_t X_sample_stride
    cdef SIZE_t X_fx_stride
    cdef DOUBLE_t* y
    cdef SIZE_t y_stride
    cdef DOUBLE_t* sample_weight

    cdef DOUBLE_t* data
    cdef SIZE_t* indices
    cdef SIZE_t* indptr
    cdef SIZE_t data_stride
    cdef SIZE_t indices_stride
    cdef SIZE_t indptr_stride

    # The samples vector `samples` is maintained by the Splitter object such
    # that the samples contained in a node are contiguous. With this setting,
    # split reorganizes the node samples `samples[start:end]` in two
    # subsets `samples[start:pos]` and `samples[pos:end]`.
    # The 1-d  `features` array of size n_features contains the features
    # indices and allows fast sampling without replacement of features.

    # The 1-d `constant_features` array of size n_features holds in
    # `constant_features[:n_constant_features]` the feature ids with
    # constant values for all the samples that reached a specific node.
    # The value `n_constant_features` is given by the the parent node to its
    # child nodes.  The content of the range `[n_constant_features:]` is left
    # undefined, but preallocated for performance reasons
    # This allows optimisation with depth-based tree building.

    # Methods
    cdef void init(self, np.ndarray X,
                         np.ndarray y,
                         DOUBLE_t* sample_weight)

    cdef void init_sparse(self, np.ndarray y,
                          np.ndarray data,
                          np.ndarray indices,
                          np.ndarray indptr,
                          DOUBLE_t* sample_weight, int number_of_features)

    cdef void node_reset(self, SIZE_t start, SIZE_t end) nogil

    cdef void node_split(self, double impurity,  # Impurity of the node
                               SIZE_t* pos, # Set to >= end if the node is a leaf
                               SIZE_t* feature,
                               double* threshold,
                               double* impurity_left,
                               double* impurity_right,
                               double* impurity_improvement,
                               SIZE_t* n_constant_features) nogil

    cdef void node_split_sparse(self, double impurity, SIZE_t* pos,
                                SIZE_t* feature, double* threshold,
                                double* impurity_left, double* impurity_right,
                                double* impurity_improvement,
                                SIZE_t* n_constant_features,
                                SIZE_t number_of_features) nogil

    cdef void node_value(self, double* dest) nogil

# =============================================================================
# Tree
# =============================================================================

cdef struct Node:
    # The main storage for Tree, excluding values at each node, which are
    # stored separately as their size is not known at compile time.
    # An array of each field is publicly accessible from Tree, and its
    # semantics is documented there.
    SIZE_t left_child
    SIZE_t right_child
    SIZE_t feature
    DOUBLE_t threshold
    DOUBLE_t impurity
    SIZE_t n_samples
    DOUBLE_t weighted_n_samples

cdef class Tree:
    # Input/Output layout
    cdef public SIZE_t n_features        # Number of features in X
    cdef SIZE_t* n_classes               # Number of classes in y[:, k]
    cdef public SIZE_t n_outputs         # Number of outputs in y
    cdef public SIZE_t max_n_classes     # max(n_classes)

    # Parameters
    cdef public Splitter splitter        # Splitting algorithm
    cdef public SIZE_t max_depth         # Max depth of the tree
    cdef public SIZE_t min_samples_split # Minimum number of samples in an internal node
    cdef public SIZE_t min_samples_leaf  # Minimum number of samples in a leaf
    cdef public object random_state      # Random state
    cdef public int max_leaf_nodes       # Number of leafs to grow

    # Inner structures: values are stored separately from node structure,
    # since size is determined at runtime.
    cdef public SIZE_t node_count        # Counter for node IDs
    cdef public SIZE_t capacity          # Capacity of tree, in terms of nodes
    cdef Node* nodes                     # Array of nodes
    cdef double* value                   # (capacity, n_outputs, max_n_classes) array of values
    cdef SIZE_t value_stride             # = n_outputs * max_n_classes

    # Methods
    cdef SIZE_t _add_node(self, SIZE_t parent,
                                bint is_left,
                                bint is_leaf,
                                SIZE_t feature,
                                double threshold,
                                double impurity,
                                SIZE_t n_node_samples,
                                double weighted_n_samples) nogil
    cdef void _resize(self, SIZE_t capacity)
    cdef int _resize_c(self, SIZE_t capacity=*) nogil

    cdef np.ndarray _get_value_ndarray(self)
    cdef np.ndarray _get_node_ndarray(self)

    cpdef np.ndarray predict(self, np.ndarray[DTYPE_t, ndim=2] X)
    cpdef np.ndarray apply(self, np.ndarray[DTYPE_t, ndim=2] X)
    cpdef compute_feature_importances(self, normalize=*)

# =============================================================================
# Tree builder
# =============================================================================

cdef class TreeBuilder:
     cpdef build(self, Tree tree, np.ndarray X, np.ndarray y,
                 np.ndarray sample_weight=*)

cdef class DepthFirstTreeBuilder(TreeBuilder):
     cpdef build(self, Tree tree, np.ndarray X, np.ndarray y,
                 np.ndarray sample_weight=*)
     cpdef build_sparse(self, Tree tree,  np.ndarray data, np.ndarray indices,
                       np.ndarray indptr, SIZE_t number_of_features,
                       np.ndarray y, np.ndarray sample_weight)

cdef class BestFirstTreeBuilder(TreeBuilder):
     cpdef build(self, Tree tree, np.ndarray X, np.ndarray y,
                 np.ndarray sample_weight=*)

     cpdef build_sparse(self, Tree tree,  np.ndarray data, np.ndarray indices,
                       np.ndarray indptr, SIZE_t number_of_features,
                       np.ndarray y, np.ndarray sample_weight)
