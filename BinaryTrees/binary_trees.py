# binary_trees.py
"""Volume 2: Binary Trees.
Caelan Osman
Math 321 sec 3
October 8, 2020
"""

# These imports are used in BST.draw().
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib import pyplot as plt
import random
import time


class SinglyLinkedListNode:
    """A node with a value and a reference to the next node."""
    def __init__(self, data):
        self.value, self.next = data, None

class SinglyLinkedList:
    """A singly linked list with a head and a tail."""
    def __init__(self):
        self.head, self.tail = None, None

    def append(self, data):
        """Add a node containing the data to the end of the list."""
        n = SinglyLinkedListNode(data)
        if self.head is None:
            self.head, self.tail = n, n
        else:
            self.tail.next = n
            self.tail = n

    def iterative_find(self, data):
        """Search iteratively for a node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        current = self.head
        while current is not None:
            if current.value == data:
                return current
            current = current.next
        raise ValueError(str(data) + " is not in the list")

    # Problem 1
    def recursive_find(self, data):
        """This function searches recursively for the node containing the data.
        Returns:
            (SinglyLinkedListNode): the node containing the data.
        raises:
            (ValueError): If the list is empty or the data is not contained in the list. 
        """
        #function to revursively find node
        def is_node(node):
            #if the SLL is empty raise a value error
            if node == None:
                raise ValueError('The list is empty')
            #if current node contains the data return that node
            elif node.value == data:
                return node
            #if the next node is none then we know we are at the end of our list and no other node
            #contained the value so we can rais a value error that the data isn't contained in the SLL
            elif node.next == None:
                raise ValueError('The data is not contained in the list')
            #recursive call on the next node in the lsit
            else:
                return is_node(node.next)

        #initial call on the head node in the SLL
        return is_node(self.head)

class BSTNode:
    """A node class for binary search trees. Contains a value, a
    reference to the parent node, and references to two child nodes.
    """
    def __init__(self, data):
        """Construct a new node and set the value attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.value = data
        self.prev = None        # A reference to this node's parent node.
        self.left = None        # self.left.value < self.value
        self.right = None       # self.value < self.right.value


class BST:
    """Binary search tree data structure class.
    The root attribute references the first node in the tree.
    """
    def __init__(self):
        """Initialize the root attribute."""
        self.root = None

    def find(self, data):
        """Return the node containing the data. If there is no such node
        in the tree, including if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing
            the data is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree.")
            if data == current.value:               # Base case 2: data found!
                return current
            if data < current.value:                # Recursively search left.
                return _step(current.left)
            else:                                   # Recursively search right.
                return _step(current.right)

        # Start the recursion on the root of the tree.
        return _step(self.root)

    # Problem 2
    def insert(self, data):
        """Insert a new node containing the specified data.

        Raises:
            ValueError: if the data is already in the tree.
        """
        #recurssive function to step through BST and insert node where we want it to be
        def _step(node):
            if data < node.value: #if the data is less than the node value we insert to the left
                if node.left == None: #if there is no left node insert the data here
                    new = BSTNode(data)
                    node.left = new
                    new.prev = node
                else: #otherwise recursive call to _step(node) function
                    _step(node.left)

            #if the data is grater than the node value do the same thing about except on the right.
            #notice node.value != data because of the try except block below
            else:
                if node.right == None:
                    new = BSTNode(data)
                    node.right = new
                    new.prev = node
                else:
                    _step(node.right)

        try: #try to find the data
            self.find(data)
        except ValueError: #if the data is not found insert our new node
            if self.root == None: #if there is no root, the BST is empty, make the new node the root
                self.root = BSTNode(data)
            else: #otherwise call the step function
                _step(self.root)
        else: #if the data is found raise a value error about duplicates
            raise ValueError('Data already contained in tree, no duplicates allowed')

    # Problem 3
    def remove(self, data):
        """This function removes the node containing the specified data.
        Raises:
            ValueError: if there is no node containing the data, including if
                the tree is empty.
        """
        #if the BST is empty raise value error
        if self.root == None:
           raise ValueError('The BST is empty.')

        #this is the node to remove; self.find(data) will raise a value error
        #if there is no node in the BST containing the data
        node_to_remove = self.find(data)

        if node_to_remove.left == None and node_to_remove.right == None: #if the node to remove is a leaf node
            if node_to_remove.prev == None: #if it is the root node
                self.root = None
            elif node_to_remove.prev.left == node_to_remove: #if it is a left child of the parent node set the left pointer to None
                node_to_remove.prev.left = None
            else: #otherwise it is a right child so set the parents right child to none
                node_to_remove.prev.right = None
        elif node_to_remove.right == None: #if the node to remove has only a left child
            if self.root == node_to_remove: #if this node happens to be the root
                self.root = self.root.left
                self.root.prev = None
            elif node_to_remove.prev.left == node_to_remove: #if the node to remove is a left child of its parent node
                prev_node = node_to_remove.prev
                prev_node.left = node_to_remove.left
                prev_node.left.prev = prev_node
            else: #otherwise it is a right child of the parent node
                prev_node = node_to_remove.prev
                prev_node.right = node_to_remove.left
                prev_node.right.prev = prev_node
        elif node_to_remove.left == None: #if the node to remove has only a right child
            if self.root == node_to_remove: #if this node happens to be the root
                self.root = self.root.right
                self.root.prev = None
            elif node_to_remove.prev.left == node_to_remove: #if the node to remove is a left child of its parent node
                prev_node = node_to_remove.prev
                prev_node.left = node_to_remove.right
                prev_node.left.prev = prev_node
            else: #otherwise it is a right child of the parent node
                prev_node = node_to_remove.prev
                prev_node.right = node_to_remove.right
                prev_node.right.prev = prev_node
        else: #If the node to remove has two children or is the root node
            if self.root == node_to_remove: #if the node to remove is the root node (and has two children)
                leaf = self.root.left
                while(leaf.right != None): #go to the farthest right leaf node after going one level left
                    leaf = leaf.right
                new_value = leaf.value
                self.remove(leaf.value) #remove the leaf
                self.root.value = new_value #set the roots new value
            else: #if the node to remove is not the root, follow the same strategy as above
                leaf = node_to_remove.left
                while (leaf.right != None):
                    leaf = leaf.right
                new_value = leaf.value
                self.remove(leaf.value)
                node_to_remove.value = new_value


    def __str__(self):
        """String representation: a hierarchical view of the BST.

        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed
                (2) (5)    [2, 5]       by depth levels. Edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """
        if self.root is None:                       # Empty tree
            return "[]"
        out, current_level = [], [self.root]        # Nonempty tree
        while current_level:
            next_level, values = [], []
            for node in current_level:
                values.append(node.value)
                for child in [node.left, node.right]:
                    if child is not None:
                        next_level.append(child)
            out.append(values)
            current_level = next_level
        return "\n".join([str(x) for x in out])

    def draw(self):
        """Use NetworkX and Matplotlib to visualize the tree."""
        if self.root is None:
            return

        # Build the directed graph.
        G = nx.DiGraph()
        G.add_node(self.root.value)
        nodes = [self.root]
        while nodes:
            current = nodes.pop(0)
            for child in [current.left, current.right]:
                if child is not None:
                    G.add_edge(current.value, child.value)
                    nodes.append(child)

        # Plot the graph. This requires graphviz_layout (pygraphviz).
        nx.draw(G, pos=graphviz_layout(G, prog="dot"), arrows=True,
                with_labels=True, node_color="C1", font_size=8)
        plt.show()

class AVL(BST):
    """Adelson-Velsky Landis binary search tree data structure class.
    Rebalances after insertion when needed.
    """
    def insert(self, data):
        """Insert a node containing the data into the tree, then rebalance."""
        BST.insert(self, data)      # Insert the data like usual.
        n = self.find(data)
        while n:                    # Rebalance from the bottom up.
            n = self._rebalance(n).prev

    def remove(*args, **kwargs):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() is disabled for this class")

    def _rebalance(self,n):
        """Rebalance the subtree starting at the specified node."""
        balance = AVL._balance_factor(n)
        if balance == -2:                                   # Left heavy
            if AVL._height(n.left.left) > AVL._height(n.left.right):
                n = self._rotate_left_left(n)                   # Left Left
            else:
                n = self._rotate_left_right(n)                  # Left Right
        elif balance == 2:                                  # Right heavy
            if AVL._height(n.right.right) > AVL._height(n.right.left):
                n = self._rotate_right_right(n)                 # Right Right
            else:
                n = self._rotate_right_left(n)                  # Right Left
        return n

    @staticmethod
    def _height(current):
        """Calculate the height of a given node by descending recursively until
        there are no further child nodes. Return the number of children in the
        longest chain down.
                                    node | height
        Example:  (c)                  a | 0
                  / \                  b | 1
                (b) (f)                c | 3
                /   / \                d | 1
              (a) (d) (g)              e | 0
                    \                  f | 2
                    (e)                g | 0
        """
        if current is None:     # Base case: the end of a branch.
            return -1           # Otherwise, descend down both branches.
        return 1 + max(AVL._height(current.right), AVL._height(current.left))

    @staticmethod
    def _balance_factor(n):
        return AVL._height(n.right) - AVL._height(n.left)

    def _rotate_left_left(self, n):
        temp = n.left
        n.left = temp.right
        if temp.right:
            temp.right.prev = n
        temp.right = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_right_right(self, n):
        temp = n.right
        n.right = temp.left
        if temp.left:
            temp.left.prev = n
        temp.left = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_left_right(self, n):
        temp1 = n.left
        temp2 = temp1.right
        temp1.right = temp2.left
        if temp2.left:
            temp2.left.prev = temp1
        temp2.prev = n
        temp2.left = temp1
        temp1.prev = temp2
        n.left = temp2
        return self._rotate_left_left(n)

    def _rotate_right_left(self, n):
        temp1 = n.right
        temp2 = temp1.left
        temp1.left = temp2.right
        if temp2.right:
            temp2.right.prev = temp1
        temp2.prev = n
        temp2.right = temp1
        temp1.prev = temp2
        n.right = temp2
        return self._rotate_right_right(n)


# Problem 4
def prob4():
    """This function compares the build and search times of the SLL, BST, and AVL classes
    """

    infile = 'english.txt'
    with open(infile, 'r') as ip: #add lines to list
        data = ip.readlines()
    num_elements = [2**n for n in range(3, 11)] #number of random elements
    #sll times
    sll_load_times = []
    sll_find_times = []
    #bst times
    bst_load_times = []
    bst_find_times = []
    #avl times
    avl_load_times = []
    avl_find_times = []

    for n in num_elements: #computing times
        random_elements = random.sample(data, n) #List containing n random elements from data
        subset = random.sample(random_elements, 5) #subset containing 5 random elements from random_elements


        #timing creation times
        #sll
        sll = SinglyLinkedList()
        sll_start = time.time()
        for element in random_elements:
            sll.append(element)
        sll_end = time.time()
        sll_load_times.append(sll_end - sll_start)

        #bst
        bst = BST()
        bst_start = time.time()
        for element in random_elements:
            bst.insert(element)
        bst_end = time.time()
        bst_load_times.append(bst_end - bst_start)

        #avl
        avl = AVL()
        avl_start = time.time()
        for element in random_elements:
            avl.insert(element)
        avl_end = time.time()
        avl_load_times.append(avl_end - avl_start)



        #timing finding times
        #sll
        sll_start = time.time()
        for element in subset:
            sll.iterative_find(element)
        sll_end = time.time()
        sll_find_times.append(sll_end - sll_start)

        #bst
        bst_start = time.time()
        for element in subset:
            bst.find(element)
        bst_end = time.time()
        bst_find_times.append(bst_end - bst_start)

        #avl
        avl_start = time.time()
        for element in subset:
            avl.find(element)
        avl_end = time.time()
        avl_find_times.append(avl_end - avl_start)

    #plotting
    ax1 = plt.subplot(121)
    ax1.loglog(num_elements, sll_load_times, basex=2, basey=2, label='SLL Load Times')
    ax1.loglog(num_elements, bst_load_times, basex=2, basey=2, label='BST Load Times')
    ax1.loglog(num_elements, avl_load_times, basex=2, basey=2, label='AVL Load Times')
    ax1.legend(loc='upper left')
    plt.title('Load Times')
    plt.xlabel('Number of Elements in Structure')
    plt.ylabel('Toad Times')
    ax2 = plt.subplot(122)
    ax2.plot(num_elements, sll_find_times, label='SLL Find Times')
    ax2.plot(num_elements, bst_find_times, label='BST Find Times')
    ax2.plot(num_elements, avl_find_times, label='AVL Find Times')
    ax2.legend(loc='upper left')
    plt.title('Find Times')
    plt.xlabel('Number of Elements in Structure')
    plt.ylabel('Find Times')
    plt.show()
