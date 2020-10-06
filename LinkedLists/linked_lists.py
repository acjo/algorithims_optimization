# linked_lists.py
"""Volume 2: Linked Lists.
Caelan Osman
Math 345 Sec 3
October 3rd, 2020
"""
'''

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.find('b')
            >>> node.value
            'b'
            >>> l.find('f')
            ValueError: <message>

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.get(3)
            >>> node.value
            'd'
            >>> l.get(5)
            IndexError: <message>
'''


# Problem 1
class Node:
    """A basic node class for storing data."""
    def __init__(self, data):
        """Store the data in the value attribute.
        Raises:
            TypeError: if data is not of type int, float, or str.
        """
        if type(data) != int and type(data) != str and type(data) != float:
            raise TypeError('data is not of type "str", "int", or "float"')
        else:
            self.value = data

class LinkedListNode(Node):
    """A node class for doubly linked lists. Inherits from the Node class.
    Contains references to the next and previous nodes in the linked list.
    """
    def __init__(self, data):
        """Store the data in the value attribute and initialize
        attributes for the next and previous nodes in the list.
        """
        Node.__init__(self, data)       # Use inheritance to set self.value.
        self.next = None                # Reference to the next node.
        self.prev = None                # Reference to the previous node.


# Problems 2-5
class LinkedList:
    """Doubly linked list data structure class.

    Attributes:
        head (LinkedListNode): the first node in the list.
        tail (LinkedListNode): the last node in the list.
    """
    def __init__(self):
        """Initialize the head and tail attributes by setting
        them to None, since the list is empty initially.
        """
        self.head = None
        self.tail = None
        self.length = 0

    def append(self, data):
        """Append a new node containing the data to the end of the list."""
        # Create a new node to store the input data.
        new_node = LinkedListNode(data)
        self.length += 1
        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
        else:
            # If the list is not empty, place new_node after the tail.
            self.tail.next = new_node               # tail --> new_node
            new_node.prev = self.tail               # tail <-- new_node
            # Now the last node in the list is new_node, so reassign the tail.
            self.tail = new_node

    # Problem 2
    def find(self, data):
        """Return the first node in the list containing the data.

        Raises:
            ValueError: if the list does not contain the data.
        """
        #edge case if there are no elements in the liked list then obviously no elements can be found
        if self.length == 0:
            raise ValueError('Data is not contained in Liked List')

        #set intial conditions
        foundnode = None
        n = self.head
        while(foundnode is None): #while our node is not found
            if data == n.value: #if data is equal to the node value then return that node
                return n
            elif n == self.tail: #if we are at the last node, exit the loop
                foundnode = False
            else:
                n = n.next #if we aren't at the last, and the data hasn't been found go onto the next node

        #if the node is not found then this error is raised
        raise ValueError('Data is not contained in Linked List')

    # Problem 2
    def get(self, i):
        #if the index is larger than the size of the list through an out of range error
        if i < 0 or i >= self.length:
            raise IndexError('Index out of range')
        #otherwise get the node
        else:
            n = self.head
            if i == 0:
                return n
            n = n.next
            for j in range(0, i):
                if j == i-1:
                    return n
                else:
                    n = n.next

    # Problem 3
    def __len__(self):
        """Returns the number of nodes in the list.
        """
        return self.length

    # Problem 3
    def __str__(self):
        """This function returns a string representation of the liked list
           which is the same as the standard python list
        """

        n = self.head #starting node
        string_rep = '[' #initial string representation

        if self.length == 0: #returns an "empty list" if the length is zero
            return string_rep + ']'

        for i in range(0, self.length): #add all nodes to our representation
            if type(n.value) == str: #if the node type is string then we need to represent it as such
                if i == 0:
                    string_rep += repr(n.value)
                elif i == self.length - 1:
                    string_rep += ', ' + repr(n.value) +']'
                else:
                    string_rep += ', ' + repr(n.value)
            else:# if the node type is an int or float
                if i == 0:
                    string_rep += str(n.value)
                elif i == self.length - 1:
                    string_rep += ', ' + str(n.value) +']'
                else:
                    string_rep += ', ' + str(n.value)
            n = n.next #set our node to the next node in the linked list
        return string_rep

    # Problem 4
    def remove(self, data):
        """Remove the first node in the list containing the data.

        Raises:
            ValueError: if the list is empty or does not contain the data.

        Examples:
            >>> print(l1)               |   >>> print(l2)
            ['a', 'e', 'i', 'o', 'u']   |   [2, 4, 6, 8]
            >>> l1.remove('i')          |   >>> l2.remove(10)
            >>> l1.remove('a')          |   ValueError: <message>
            >>> l1.remove('u')          |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.remove(10)
            ['e', 'o']                  |   ValueError: <message>
        """
        raise NotImplementedError("Problem 4 Incomplete")

    # Problem 5
    def insert(self, index, data):
        """Insert a node containing data into the list immediately before the
        node at the index-th location.

        Raises:
            IndexError: if index is negative or strictly greater than the
                current number of nodes.

        Examples:
            >>> print(l1)               |   >>> len(l2)
            ['b']                       |   5
            >>> l1.insert(0, 'a')       |   >>> l2.insert(6, 'z')
            >>> print(l1)               |   IndexError: <message>
            ['a', 'b']                  |
            >>> l1.insert(2, 'd')       |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.insert(1, 'a')
            ['a', 'b', 'd']             |   IndexError: <message>
            >>> l1.insert(2, 'c')       |
            >>> print(l1)               |
            ['a', 'b', 'c', 'd']        |
        """
        raise NotImplementedError("Problem 5 Incomplete")


# Problem 6: Deque class.


# Problem 7
def prob7(infile, outfile):
    """Reverse the contents of a file by line and write the results to
    another file.

    Parameters:
        infile (str): the file to read from.
        outfile (str): the file to write to.
    """
    raise NotImplementedError("Problem 7 Incomplete")


