# breadth_first_search.py
"""Volume 2: Breadth-First Search.
Caelan Osman
Math 321, Sec 3
October 28, 2020
"""

from collections import deque

# Problems 1-3
class Graph:
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a set of
    the corresponding node's neighbors.

    Attributes:
        d (dict): the adjacency dictionary of the graph.
    """
    def __init__(self, adjacency={}):
        """Store the adjacency dictionary as a class attribute"""
        self.d = dict(adjacency)

    def __str__(self):
        """String representation: a view of the adjacency dictionary."""
        return str(self.d)

    # Problem 1
    def add_node(self, n):
        """Add n to the graph (with no initial edges) if it is not already
        present.

        Parameters:
            n: the label for the new node.
        """
        #if n is not in the set of keys add our new node
        if n not in self.d.keys():
            edges = set()
            self.d.update({n: edges})

    # Problem 1
    def add_edge(self, u, v):
        """Add an edge between node u and node v. Also add u and v to the graph
        if they are not already present.

        Parameters:
            u: a node label.
            v: a node label.
        """
        #add node
        self.add_node(u)
        self.add_node(v)

        #get the nodes adjacent to u and v and add v and u respectively
        u_edges = self.d.get(u)
        u_edges.add(v)
        v_edges = self.d.get(v)
        v_edges.add(u)

        #update the dictionary
        self.d.update({u: u_edges})
        self.d.update({v: v_edges})

    # Problem 1
    def remove_node(self, n):
        """Remove n from the graph, including all edges adjacent to it.

        Parameters:
            n: the label for the node to remove.

        Raises:
            KeyError: if n is not in the graph.
        """
        #if n is not a key value raise an error
        if n not in self.d.keys():
            raise KeyError('The node ' + str(n) + ' is not in the graph')
        #otherwise pop it from the dictionary
        else:
            self.d.pop(n)

    # Problem 1
    def remove_edge(self, u, v):
        """Remove the edge between nodes u and v.

        Parameters:
            u: a node label.
            v: a node label.

        Raises:
            KeyError: if u or v are not in the graph, or if there is no
                edge between u and v.
        """
        #if u or v are not key values raise an error
        if u not in self.d.keys() or v not in self.d.keys():
            raise KeyError('Either the node ' + str(u) + ' or ' + str(v) + ' is not a keyvalue')
        #otherwise remove the edge
        else:
            u_edges = self.d.get(u)
            v_edges = self.d.get(v)
            u_edges.remove(v)
            v_edges.remove(u)
            self.d.update({u: u_edges})
            self.d.update({v: v_edges})

    # Problem 2
    def traverse(self, source):
        """Traverse the graph with a breadth-first search until all nodes
        have been visited. Return the list of nodes in the order that they
        were visited.

        Parameters:
            source: the node to start the search at.

        Returns:
            (list): the nodes in order of visitation.

        Raises:
            KeyError: if the source node is not in the graph.
        """
        #checking to see if the source is in the graph
        if source not in self.d.keys():
            raise KeyError('The source node ' + str(source) + ' is not in the graph')
        #otherwise traverse the graph
        else:
            #to be visited
            Q = deque(source)
            #marked
            M = set(source)
            #nodes that have been visited in visitation order
            V = []

            #while Q is nonempty
            while Q:

                #pop the node off of the deque
                current = Q.pop()

                #append it to visited nodes
                V.append(current)

                #get the neighbors of current
                neighbors = self.d.get(current)

                #check if the nneighbor nodes are in marked if not, add them to Q and M,
                #add them to Q the opposite way you remove them
                for node in neighbors:
                    if node not in M:
                        Q.appendleft(node)
                        M.add(node)

            return V



    # Problem 3
    def shortest_path(self, source, target):
        """Begin a BFS at the source node and proceed until the target is
        found. Return a list containing the nodes in the shortest path from
        the source to the target, including endoints.

        Parameters:
            source: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from source to target,
                including the endpoints.

        Raises:
            KeyError: if the source or target nodes are not in the graph.
        """
        raise NotImplementedError("Problem 3 Incomplete")

data = {'A': {'B', 'D'}, 'B': {'A', 'D'}, 'C': {'D'}, 'D': {'A', 'B', 'C'}}
graph = Graph(data)
print(graph.d)
graph.add_edge('A', 'E')
print(graph.traverse('B'))


# Problems 4-6
class MovieGraph:
    """Class for solving the Kevin Bacon problem with movie data from IMDb."""

    # Problem 4
    def __init__(self, filename="movie_data.txt"):
        """Initialize a set for movie titles, a set for actor names, and an
        empty NetworkX Graph, and store them as attributes. Read the speficied
        file line by line, adding the title to the set of movies and the cast
        members to the set of actors. Add an edge to the graph between the
        movie and each cast member.

        Each line of the file represents one movie: the title is listed first,
        then the cast members, with entries separated by a '/' character.
        For example, the line for 'The Dark Knight (2008)' starts with

        The Dark Knight (2008)/Christian Bale/Heath Ledger/Aaron Eckhart/...

        Any '/' characters in movie titles have been replaced with the
        vertical pipe character | (for example, Frost|Nixon (2008)).
        """
        raise NotImplementedError("Problem 4 Incomplete")

    # Problem 5
    def path_to_actor(self, source, target):
        """Compute the shortest path from source to target and the degrees of
        separation between source and target.

        Returns:
            (list): a shortest path from source to target, including endpoints and movies.
            (int): the number of steps from source to target, excluding movies.
        """
        raise NotImplementedError("Problem 5 Incomplete")

    # Problem 6
    def average_number(self, target):
        """Calculate the shortest path lengths of every actor to the target
        (not including movies). Plot the distribution of path lengths and
        return the average path length.

        Returns:
            (float): the average path length from actor to target.
        """
        raise NotImplementedError("Problem 6 Incomplete")
