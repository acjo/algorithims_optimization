# linked_lists.py
"""Volume 2: Linked Lists.
Caelan Osman
Math 345 Sec 3
October 3rd, 2020
"""


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
        #edge case if head is None then the list is empty and the linked list cannot contain the node.
        if self.head == None:
            raise ValueError('The node is not contained in the Linked list')

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
            raise IndexError('"Index" out of range')
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
        elif self.length == 1: #handles the case in which the length of the linked list is just one
            if type(n.value) == str:
                 string_rep += repr(n.value) + ']'
                 return string_rep
            else:
               string_rep += str(n.value) + ']'
               return string_rep

        #only executes if the above if-else statement is never entered
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
        """Removes the first node in the list containing the data.

        Raises:
            ValueError: if the list is empty or does not contain the data.
        """

        if self.length == 0: #can't take something out of an empty list
            raise ValueError('The list is empty')

        n = self.head
        if self.length == 1 and data == self.head.value: #special case where the list only has one element
            self.head = None
            self.tail = None
            self.length = 0
            return
        elif data == n.value: #special case where the len(llist) > 1 and the data is the first node
            self.head = n.next
            self.head.prev = None
            self.length -= 1
            return
        elif data == self.tail.value: #special case where len(llist) > 1 and the data is the final node
            self.tail = self.tail.prev
            self.tail.next = None
            self.length -= 1
            return
        else:#finally we enter this case when none of the above apply
            n = n.next
            for i in range(1, self.length-1): #only need to loop from the 2nd to n-1 nodes because of ^^
                if data == n.value:
                    n.prev.next = n.next #next of the previous and the previous of the next
                    n.next.prev = n.prev
                    self.length -= 1
                    return
                else: #otherwise loop onto the next
                    n = n.next

        raise ValueError('Data not found') #finally if the data is not found

    # Problem 5
    def insert(self, index, data):
        """Insert a node containing data into the list immediately before the
        node at the index-th location.

        Raises:
            IndexError: if index is negative or strictly greater than the
                current number of nodes.
        """
        if index < 0 or index > self.length: #throws an out of range error if the index doesn't make sense
            raise IndexError('"Index" Out of range')
        elif index == 0 and self.length > 0: #special case where we are inserting and resetting the head and the list is not empty
            new = LinkedListNode(data) #construct new node
            new.next = self.head #set the pointer to the current head
            new.next.prev = new  #set the pointer of the next back to the new
            self.head = new #set the head as the new
            self.length += 1 #update length
            return
        elif index == 0 and self.length == 0: #Special case where we are inserting and resetting the head and the list is empty
            new = LinkedListNode(data)
            self.head = new
            self.tail = new
            self.length = 1
        elif index == self.length: #if the insert is the length of the list just append
            self.append(data)
        else: #do the same as in the first elif statement except at an index that is greater than one
            n = self.head
            for j in range(0,index-1):
                n = n.next
            new = LinkedListNode(data)
            new.prev = n
            new.next = n.next
            n.next = new
            new.next.prev = new
            self.length += 1
            return

class Deque(LinkedList):
    '''Deque data structure class:

       Attributes:
           Head Node
           Tail Node
           length of deque
    '''

    #use LinkedList class constructor
    def __init__(self):
        LinkedList.__init__(self)

    def pop(self): #function to remove last node
        if self.head == None: #edge case if deque is empty
            raise ValueError('The deque is empty, cannot pop.')
        elif self.length == 1: #edge case if deque only has one node
            node_value = self.tail.value
            self.tail = None
            self.head = None

            self.length = 0
            return node_value
        else: #otherwise reset tail value
            node_value = self.tail.value
            self.tail = self.tail.prev
            self.tail.next = None
            self.length -= 1
            return node_value

    def popleft(self): #function to remove first node
        if self.head == None: #edge case if our deque is emtpy
            raise ValueError('The deque is empty, cannot popleft.')
        elif self.length == 1: #edge case if our deque only has one element
            node_value = self.head.value
            self.tail = None
            self.head = None
            self.length = 0
            return node_value
        else: #otherwise reset the head value uses inheritance and the remove function from the linked list class
            node_value = self.head.value
            LinkedList.remove(self, node_value)
            return node_value

    def appendleft(self, data): #if we want to add nodes to the beginning of our deque we just use inheritence on the insert function with index 0.
        LinkedList.insert(self, 0, data)

    #these just remove functionality that we don't want to be able to be accessed because deques are restricted access
    def remove(*args, **kwargs):
        raise NotImplementedError('Use pop() or popleft() for removal')

    def insert(*args, **kwargs):
        raise NotImplementedError('Use append() or appendleft()')

# Problem 7
def prob7(infile, outfile):
    """Reverse the contents of a file by line and write the results to
    another file.

    Parameters:
        infile (str): the file to read from.
        outfile (str): the file to write to.
    """
    filedeque = Deque()
    with open(infile, 'r') as ip:
        lines = ip.readlines()
    for i in range(0, len(lines)):
        if i == len(lines) - 1:
            filedeque.appendleft(lines[i] + '\n')
        else:
            filedeque.appendleft(lines[i])
    with open(outfile, 'w') as of:
        for i in range(0, filedeque.length):
            of.write(filedeque.popleft())

























