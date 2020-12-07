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
        Node.__init__(self, data) # Use inheritance to set self.value.
        self.next = None # Reference to the next node.
        self.prev = None # Reference to the previous node.


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
        new_node = LinkedListNode(data) # Create a new node to store the input data.
        self.length += 1
        if self.head is None: # If the list is empty, assign the head and tail attributes to
            self.head = new_node
            self.tail = new_node
        else:
            # If the list is not empty, place new_node after the tail.
            self.tail.next = new_node # tail --> new_node
            new_node.prev = self.tail # tail <- new_node
            self.tail = new_node #reassign the tail

    # Problem 2
    def find(self, data):
        """Return the first node in the list containing the data.

        Raises:
            ValueError: if the list does not contain the data.
        """
        #edge case if the list is empty head is None then the list is empty and the linked list cannot contain the node.
        if self.head == None:
            raise ValueError('The node is not contained in the Linked list')

        #set intial conditions
        foundnode = None
        n = self.head
        while(foundnode is None): #repeat until we find node
            if data == n.value: #if data is equal to the node value then return that node
                return n
            elif n == self.tail: #if we are at the last node, exit the loop
                foundnode = False
            else:
                n = n.next #otherwise check next node

        #if the node is not found then this error is raised
        raise ValueError('Data is not contained in Linked List')

    # Problem 2
    def get(self, i):
        #if the index is larger than the size of the list through an out of range error
        if i < 0 or i >= self.length:
            raise IndexError('"Index" out of range')

        #otherwise loop through and get the node
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

        #starting node and initial string representation
        n = self.head
        string_rep = '['

        if self.length == 0: #returns an "empty list" if the length is zero
            return string_rep + ']'
        elif self.length == 1: #if the lenght of the list is one
            if type(n.value) == str:
                 string_rep += repr(n.value) + ']'
                 return string_rep
            else:
               string_rep += str(n.value) + ']'
               return string_rep

        #if the length of the list is greater than one.
        for i in range(0, self.length): #loop through nodes
            if type(n.value) == str: #if the node type is string
                if i == 0: #deals with comma and bracket placement
                    string_rep += repr(n.value)
                elif i == self.length - 1:
                    string_rep += ', ' + repr(n.value) +']'
                else:
                    string_rep += ', ' + repr(n.value)
            else: #same as above but for numbers
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

        #if the list is empty
        if self.length == 0:
            raise ValueError('The list is empty')

        #special case where the list only has one element
        n = self.head
        if self.length == 1 and data == self.head.value:
            self.head = None
            self.tail = None
            self.length = 0
            return
        #special case where the len(llist) > 1 and the data is the first node
        elif data == n.value:
            self.head = n.next
            self.head.prev = None
            self.length -= 1
            return
        #special case where len(llist) > 1 and the data is the final node
        elif data == self.tail.value:
            self.tail = self.tail.prev
            self.tail.next = None
            self.length -= 1
            return
        else:
            n = n.next
            #only need to loop from the 2nd to n-1 nodes because of ^^
            for i in range(1, self.length-1):
                if data == n.value:
                    #next of the previous and the previous of the next
                    n.prev.next = n.next
                    n.next.prev = n.prev
                    self.length -= 1
                    return
                #otherwise loop onto the next
                else:
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
        #throws an out of range error if the index doesn't make sense
        if index < 0 or index > self.length:
            raise IndexError('"Index" Out of range')
        #special case where we are inserting and resetting the head and the list is not empty
        if index == 0 and self.length > 0:
            new = LinkedListNode(data) #construct new node
            new.next = self.head #set the pointer to the current head
            new.next.prev = new  #set the pointer of the next back to the new
            self.head = new #set the head as the new
            self.length += 1 #update length
            return
        #Special case where we are inserting and resetting the head and the list is empty
        elif index == 0 and self.length == 0:
            new = LinkedListNode(data)
            self.head = new
            self.tail = new
            self.length = 1
            return
        #if the insert is the length of the list just append
        elif index == self.length:
            self.append(data)
            return
        #do the same as in the first elif statement except at an index > 1
        else:
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
        #use inhertiance to remove node
        else:
            node_value = self.head.value
            LinkedList.remove(self, node_value)
            return node_value
    #if we want to add nodes to the beginning of our deque we just use
    #inheritence on the insert function with index 0.
    def appendleft(self, data):
        LinkedList.insert(self, 0, data)

    #these just remove functionality that we don't want to be able
    #to be accessed because deques are restricted access
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
    with open(infile, 'r') as ip_file:
        contents = ip_file.read()
        splitline = contents.strip().split('\n')
        for i in range(len(splitline)):
            filedeque.append(str(splitline[i]))

    print(filedeque)
    final_string = ''
    with open(outfile, 'w') as op_file:
        for i in range(filedeque.length):
            final_string += str(filedeque.pop()) + '\n'
            #op_file.write(str(filedeque.pop()) + '\n')

    return final_string


    '''
    final_string = ''
    filedeque = Deque() #create our deque
    with open(infile, 'r') as ip: #open our file
        lines = ip.read().strip().split('\n') #add all lines to a list
    for i, line in enumerate(lines): #populate our deque
        #if we are at the last element make sure to add another newline character
        if i == len(lines) - 1:
            filedeque.appendleft('\n' + line + '\n')
        #make sure to not add an additional newline
        elif i == 0:
            filedeque.appendleft(line)
        else:
            filedeque.appendleft(line + '\n')
    #assemble string
    for _ in range(filedeque.length):
        final_string += filedeque.popleft()
    #write to our outfile
    with open(outfile, 'w') as of:
        of.write(final_string)

    '''

if __name__ == '__main__':

    backwards_file_string = '\noutlooks\neuthanasia\noutlier\nyours\nglistened\nsociolinguistics\nfixations\ndoubts\nencyclopaedic\nleer'
    deque = Deque()
    make_lines = backwards_file_string.split('\n')
    for line in make_lines:
        deque.append(line)
    with open('test.txt', 'w') as key:
        for _ in range(deque.length):
            key.write(deque.pop() + '\n')

    string = prob7('test.txt', 'test_output.txt')
    print(repr(string))
