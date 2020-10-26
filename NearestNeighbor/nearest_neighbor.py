# nearest_neighbor.py
"""Volume 2: Nearest Neighbor Search.
Caelan Osman
Math 321 sec 3
October 25, 2020
"""

import numpy as np
from scipy import linalg as la

# Problem 1
def exhaustive_search(X, z):
    """This function solves the nearest neighbor search problem with an exhaustive search.
    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.
    Returns:
        min_vec ((k,) ndarray) the element (row) of X that is nearest to z.
        min_dist (float) The Euclidean distance from the nearest neighbor to z.
    """
    diff = X - z #use array broadcasting to find the difference of each row of X and the elements of Z
    distances = la.norm(diff, axis = 1) #an array containing all norm values of the rows
    min_dist = min(distances) #sets the minimum distanced
    find = np.where(distances == min_dist)[0][0] #find the first index where the min distance occurs
    min_vec = X[find, :] #set the min vec to the the row at find index

    return min_vec, min_dist

# Problem 2: Write a KDTNode class.
class KDTNode:
    """Node class for K-D Trees.

    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this node's right child.
        value ((k,) ndarray): a coordinate in k-dimensional space.
        pivot (int): the dimension of the value to make comparisons on.
    """
    def __init__(self, x):
        if type(x) != np.ndarray:
            raise TypeError('Input data is not of correct type')
        else: #initializing value and children
            self.value = x
            self.left = None
            self.right = None
            self.pivot = None

# Problems 3 and 4
class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    # Problem 3
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.
        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
            ValueError: if data is already in the tree
        """
        new = KDTNode(data) #takes care of the type error

        if self.root != None:
            if self.k != len(data): #raises value error if new data doesn't have the same dimension
                raise ValueError('The input data is not of the same dimension as other values in the tree')

        def _step(node):
            if data[node.pivot] < node.value[node.pivot]: #need to put it on the left
                if node.left == None: #if there is no left node insert the data here
                    #new = KDTNode(data) #create the new node
                    node.left = new #point the parent to it
                    if node.pivot == self.k - 1: #if the parents pivot is k-1 set child's pivot to 0
                        new.pivot = 0
                    else: #set child's pivot to k+1
                        new.pivot = node.pivot + 1
                else:#otherwise recursive call on the left node
                    _step(node.left)
            #if the data at the pivot is greater than or equal to the node value do the same thing
            else:
                if node.right == None:
                    #new = KDTNode(data)
                    node.right = new
                    if node.pivot == self.k - 1:
                        new.pivot = 0
                    else:
                        new.pivot = node.pivot + 1
                else:
                    _step(node.right)

        try: #try to find the data
            self.find(data)
        except ValueError: #if the data isn't in the tree
            if self.root == None: #if the tree is empty set the root node, k, and the pivot
                self.root = new
                self.root.pivot = 0
                self.k = len(data)
            else:#otherwise recursively step through starting with the root node
                _step(self.root)
        else: #if the data is found raise error
            raise ValueError('Data already contained in tree, no duplicates allowed')

        #pulled from pdf file

        def __str__(self):
            """String representation: a hierarchical list of nodes and their axes.
            Example:
                             'KDT(k=2)
               [5,5]           [5 5] pivot = 0
                / \            [3 2] pivot = 1
            [3,2] [8,4]        [8 4] pivot = 1
              \      \         [2 6] pivot = 0
            [2,6] [7,5]        [7 5] pivot = 0'
            """

            if self.root is None:
                return "Empty KDT"
            nodes, strs = [self.root], []
            while nodes:
                current = nodes.pop(0)
                strs.append("{}\tpivot = {}".format(current.value, current.pivot))
                for child in [current.left, current.right]:
                    if child:
                        nodes.append(child)
            return "KDT(k={})\n".format(self.k) + "\n".join(strs)

    # Problem 4
    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        raise NotImplementedError("Problem 4 Incomplete")

    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)

'''
tree = KDT()

to_insert = [np.array([3,1,4]), np.array([1,2,7]), np.array([4,3,5]), np.array([2, 0, 3]), np.array([2, 4, 5]), np.array([6, 1, 4]), np.array([1, 4, 3]), np.array([0, 5, 7]), np.array([5,2,5])]

for node in to_insert:
    tree.insert(node)

node = tree.find(np.array([5,2,5]))
print(node.value)
print(node.left)
print(node.right)
print(node.pivot)
'''

# Problem 5: Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    """A k-nearest neighbors classifier that uses SciPy's KDTree to solve
    the nearest neighbor problem efficiently.
    """

# Problem 6
def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    raise NotImplementedError("Problem 6 Incomplete")
