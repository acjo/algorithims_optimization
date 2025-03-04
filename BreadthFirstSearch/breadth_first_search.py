# breadth_first_search.py
"""Volume 2: Breadth-First Search.
Caelan Osman
Math 321, Sec 3
October 28, 2020
"""

from collections import deque
import networkx as nx
from matplotlib import pyplot as plt
import time

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
        #add nodes
        self.add_node(u)
        self.add_node(v)

        #add v to the nodes adjacent to u and visa vers
        self.d[u].add(v)
        self.d[v].add(u)

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
        #and remove it from all adjacent nodes
        else:
            for node in self.d.keys():
                self.d[node].discard(n)
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
            self.d[v].discard(u)
            self.d[u].discard(v)

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
            Q = deque()
            Q.append(source)
            #marked
            M = set(source)
            #nodes that have been visited in visitation order
            V = []

            #while Q is nonempty
            while Q:
                #pop the node off of the deque
                current = Q.popleft()
                #append it to visited nodes
                V.append(current)
                #get the neighbors of current
                #add them to Q the opposite way you remove them
                for node in self.d[ current ]:
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
        if source not in self.d.keys() or target not in self.d.keys():
            return KeyError('Either the source or the target nodes are not in the graph')
        else:
            #got from algorithm 4.2 in the vol2 textbook
            #first partial path
            first_path = [source]
            #deque containing all partial paths
            P = deque()
            P.appendleft(first_path)
            #marked nodes
            M = set()
            M.add(source)

            #while P is nonempty
            while P:
                #pop the first path off of the list
                current = P.pop()
                #add the path's last node to M
                M.add(current[-1])

                #if the path's last node is the target node
                if current[-1] == target:
                    return current

                #difference of two sets, append the new partial path
                for node in self.d[current[-1]] - M:
                    P.appendleft(current + [node])

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

        #constructing the empty network
        self.network = nx.Graph()
        #initialize movie titls and actor names
        self.movie_titles = set()
        self.actor_names = set()

        #open the file and read line by line
        with open(filename, 'r', encoding="utf8") as infile:
            lines = infile.read().strip().split('\n')
        #add the correct content to each set
        #add the edges between nodes
        for line in lines:
            #grab content for the movie
            movie_content = line.strip().split('/')
            #grab the title
            current_title = movie_content[0]
            #add the movie title to the attribute
            self.movie_titles.add(current_title)
            #add the actors and edges to the the actor names attribute and the network attribute
            for i in range(1, len(movie_content)):
                current_actor = movie_content[i]
                self.actor_names.add(current_actor)
                self.network.add_edge(current_actor, current_title)


    # Problem 5
    def path_to_actor(self, source, target):
        """Compute the shortest path from source to target and the degrees of
        separation between source and target.

        Returns:
            (list): a shortest path from source to target, including endpoints and movies.
            (int): the number of steps from source to target, excluding movies.
        """
        #grabs the path
        path = nx.shortest_path(self.network, source, target)
        #grabs the number of steps
        steps = nx.shortest_path_length(self.network, source, target)
        return path, steps

    # Problem 6
    def average_number(self, target):
        """Calculate the shortest path lengths of every actor to the target
        (not including movies). Plot the distribution of path lengths and
        return the average path length.

        Returns:
            (float): the average path length from actor to target.
        """
        #get all distances to target
        shortest_distances = nx.shortest_path_length(self.network, target=target)

        #get rid of the distances from movies
        actor_distances = []
        for source in shortest_distances:
            if source in self.actor_names:
                actor_distances.append(shortest_distances[source] // 2)

        #compute the average
        avg = sum(actor_distances) / len(actor_distances)
        #plot the histogram
        # plt.hist(actor_distances, bins=[i-0.5 for i in range(8)])
        # plt.show()
        return avg


if __name__ == "__main__":


    # adjacency = {'A': {'B', 'D'},
    #              'B': {'A', 'D'},
    #              'C': {'D'},
    #              'D': {'A', 'B', 'C'}}

    # G = Graph( adjacency )

    # print( G.traverse( 'A' ) )

    # MV = MovieGraph( )

    start = time.time( )
    MV = MovieGraph( "BreadthFirstSearch\\movie_data_small.txt" )
    end = time.time( )
    print( "construction time:", end - start )
    print("number of nodes:", len( MV.network.nodes ) )
    print( "number of edges:", len( MV.network.edges ) )
    print( "number of unique movie titles:", len( MV.movie_titles ) )
    print( "number of unique actor names:", len( MV.actor_names ) )
    print( "what the total num of nodes should be:", len( MV.movie_titles ) + len( MV.actor_names ) )
    print( ) 

    for ( actor1, actor2 ) in [ ( "Samuel L. Jackson", "Kevin Bacon" ), ( "Ewan McGregor", "Kevin Bacon"), ("Jennifer Lawrence", "Kevin Bacon"), ( "Mark Hamill", "Kevin Bacon" ) ]:
        print( MV.path_to_actor( actor1, actor2 ) )


    start = time.time( )
    print( MV.average_number( "Kevin Bacon" ) )
    end = time.time( )

    print( end-start )