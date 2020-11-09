# markov_chains.py
"""Volume 2: Markov Chains.
Caelan Osman
Math 321 Sec. 3
Nov 8, 2020
"""

import numpy as np
from scipy import linalg as la

class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        (fill this out)
    """
    # Problem 1
    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            >>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        #check that A is square
        if A.shape[0] != A.shape[1]:
            raise ValueError('A is not a square matrix')
        #check that A is column stochastic
        elif not np.allclose(A.sum(axis=0), np.ones(A.shape[0])):
            raise ValueError('A is not column stochastic')
        #otherwise construct object
        else:
            #set the transition and labels attribute, initialize mapping to an empty dictionary
            self.transition_matrix = np.copy(A)
            self.mapping = dict()

            #update the mappings based on whether or not there are states
            if states == None:
                self.labels = [i for i in range(0, self.transition_matrix.shape[0])]
                for i in range(0, self.transition_matrix.shape[0]):
                    self.mapping.update({i:i})
            else:
                self.labels = states
                for i in range(0, self.transition_matrix.shape[0]):
                    self.mapping.update({self.labels[i]:i})

    # Problem 2
    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """
        #which state to transition to
        draw = np.random.multinomial(1, self.transition_matrix[:, self.mapping[state]])
        #the index of the transition
        index = np.argmax(draw)
        #return the label at that index
        return self.labels[index]

    # Problem 3
    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """
        #create the initial state_labels list
        state_labels = []
        state_labels.append(start)

        #add the N-1 states
        for i in range(0, N-1):
            new_state = self.transition(state_labels[-1])
            state_labels.append(new_state)

        return state_labels

    # Problem 3
    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        #initialize state path
        state_path = []
        #add the start state to it
        state_path.append(start)
        #iterate through the transitions until you reach the stopping point
        while state_path[-1] != stop:
            new_state = self.transition(state_path[-1])
            state_path.append(new_state)

        #return the path
        return state_path


    # Problem 4
    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """
        #generate random state distribution vector all elements non-negative and sum to zero
        x_old = np.random.random(self.transition_matrix.shape[0])
        x_old /= sum(x_old)

        #iterate through and
        i = 0
        while i < maxiter:
            #calculate new state vector
            x_new = self.transition_matrix @ x_old
            #calculate the difference using the one norm
            diff = np.linalg.norm((x_old - x_new), ord=1)
            #check difference is within tolerance
            if diff < tol:
                return x_new
            #update values
            x_old = x_new
            i += 1

        #if the tolerance is never met raise a ValueError
        raise ValueError('A^k does not converge')

'''
labels_1 = [ 'hot', 'cold' ]
B = np.array( [ [ 0.7, 0.6 ],
                [ 0.3, 0.4 ] ] )

m_chain_1 = MarkovChain(B, labels_1)
print(m_chain_1.steady_state(1e-15, 15))
print(type(m_chain_1.steady_state()))

'''




'''
labels =[ 'hot', 'mild', 'cold', 'freezing' ]
A = np.array( [ [ 0.5, 0.3, 0.1, 0 ],
                [ 0.3, 0.3, 0.3, 0.3 ],
                [ 0.2, 0.3, 0.4, 0.5 ],
                [ 0, 0.1, 0.2, 0.2 ] ] )

m_chain = MarkovChain(A, labels)
m_chain.steady_state()
'''

class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        (fill this out)
    """
    # Problem 5
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        raise NotImplementedError("Problem 5 Incomplete")

    # Problem 6
    def babble(self):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            >>> yoda = SentenceGenerator("yoda.txt")
            >>> print(yoda.babble())
            The dark side of loss is a path as one with you.
        """
        raise NotImplementedError("Problem 6 Incomplete")
