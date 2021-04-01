# dynamic_programming.py
"""Volume 2: Dynamic Programming.
Caelan Osman
Math 323 Sec. 2
March 30, 2021
"""

import numpy as np
from matplotlib import pyplot as plt

def calc_stopping(N):
    """Calculate the optimal stopping time and expected value for the
    marriage problem.

    Parameters:
        N (int): The number of candidates.

    Returns:
        (float): The maximum expected value of choosing the best candidate.
        (int): The index of the maximum expected value.
    """

    #initalize empty list and append 0 to it
    expected_vals = []
    expected_vals.append(0)
    #iterate top down
    for num in range(N, 0, -1):
        if num == N:
            V_prev = 0
        else:
            #find max value and append it to expected values
            V_cur = max(V_prev *  num/ (num + 1) + (1 / N), V_prev)
            expected_vals.append(V_cur)
            #set previous to current
            V_prev = V_cur

    #reverse order
    expected_vals = expected_vals[::-1]
    #get the max index
    index = np.argmax(expected_vals)

    #return expected val index and the index (+1)
    return expected_vals[index], index+1

# Problem 2
def graph_stopping_times(M):
    """Graph the optimal stopping percentage of candidates to interview and
    the maximum probability against M.

    Parameters:
        M (int): The maximum number of candidates.

    Returns:
        (float): The optimal stopping percent of candidates for M.
    """
    #get the domain for plotting
    domain = np.arange(3, M + 1)
    #intialize list of percentages and probabilities
    stopping_percentages = []
    probabilities = []
    #initialize optimal value
    optimal = 0
    for N in domain:
        #calculate stopping point
        val, t = calc_stopping(N)
        #calculate and stopping percentages
        stopping_percentages.append(t / N)
        #append probabilities
        probabilities.append(val)
        #increment optimal value
        optimal += val

    #plot
    plt.plot(domain, stopping_percentages, 'b--', label='Stopping Percentage')
    plt.plot(domain, probabilities, 'c-', label='Maximum Probability')
    plt.legend(loc='best')
    plt.show()

    #return optimal stopping percentage
    return optimal / M

# Problem 3
def get_consumption(N, u=lambda x: np.sqrt(x)):
    """Create the consumption matrix for the given parameters.

    Parameters:
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        u (function): Utility function.

    Returns:
        C ((N+1,N+1) ndarray): The consumption matrix.
    """
    #create evenly spaced pices
    w = np.linspace(0, 1, N+1)
    #initalize consumption matrix
    C = np.zeros((N+1, N+1))
    #populate consumption matrix

    for row in range(N+1):
        for col in range(row):
            C[row, col] = u(w[row] - w[col])

    return C

# Problems 4-6
def eat_cake(T, N, B, u=lambda x: np.sqrt(x)):
    """Create the value and policy matrices for the given parameters.

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        A ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            value of having w_i cake at time j.
        P ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            number of pieces to consume given i pieces at time j.
    """
    #pieces of cake
    w = np.linspace(0, 1, N+1)

    #initialize value function matrix and Policy matrix
    A = np.zeros((N+1, T+1))
    P = np.zeros_like(A)
    #populate value function  matrix
    for row in range(N+1):
        possible_values = [w[row] - w[j] for j in range(N+1)]
        index = np.argmax(possible_values)
        A[row, T] = u(possible_values[index])
        P[row, T] = possible_values[index]

    #we now finish populating the value function and policy matrices
    for t in range(T-1, -1, -1):
        #creat current value matrix
        CV = np.array([[u(w[i]-w[j]) + B * A[j, t+1] if j <= i else 0 for j in range(N+1)] for i in range(N+1)])
        #update A
        A[:, t] = np.max(CV, axis=1)
        #update policy matrix
        for row in range(N+1):
            wt = np.argmax(CV[row, :])
            P[row, t] = w[row] - w[wt]

    return A, P

# Problem 7
def find_policy(T, N, B, u=np.sqrt):
    """Find the most optimal route to take assuming that we start with all of
    the pieces. Show a graph of the optimal policy using graph_policy().

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        ((T,) ndarray): The matrix describing the optimal percentage to
            consume at each time.
    """
    #compute slice size
    slice_size = 1/N
    #compute value and policy matrices
    A, P = eat_cake(T, N, B, u)
    #initialize optimal policy vector
    optimal = np.zeros(T+1)
    #opupulate policy vector
    optimal[0] = P[N, 0]
    #get intial slices to subtract
    subtract = int(slice_size / P[N, 0])
    for i in range(1, T+1):
        #get new optimal
        optimal[i] = P[N-subtract, i]
        #update subtraction constan
        subtract += int(slice_size / optimal[i])

    return optimal

if __name__ == "__main__":

    #TODO: Ask about returning index
    #TODO: Ask about stopping percentage (optimal values don't match in prob2)
    #TODO: Ask about 18.4 (isn't u always increasing?)
    #TODO: Ask about rounding and policy
    #TODO: Ask about consumption matrix

    #prob1
    #print(np.allclose(calc_stopping(4)[0], 0.4583))
    #print(calc_stopping(4) == 1)

    #prob 2
    '''
    optimal = graph_stopping_times(1000)
    print(optimal)
    print(np.allclose(optimal, 0.368))
    '''

    #prob3
    '''
    consume = get_consumption(4, u=lambda x: x)
    print(consume)

    test = np.matrix([[0, 0, 0, 0, 0],
               [0.25, 0, 0, 0, 0],
               [0.5, 0.25, 0, 0, 0],
               [0.75, 0.5, 0.25, 0, 0],
               [1, 0.75, 0.5, 0.25, 0]])

    print(np.allclose(consume, test))
    '''

    #prob4, 5, 6
    '''
    a, p = eat_cake(3, 4, 0.9, u=lambda x: np.sqrt(x))
    print(a)
    print()
    print(p)
    '''

    #prob7
    #print(find_policy(3, 4, 0.9, u=np.sqrt))
