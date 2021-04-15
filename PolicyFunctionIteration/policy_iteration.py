# policy_iteration.py
"""Volume 2: Policy Function Iteration.
Caelan Osman
Math 323 Sec. 2
April 3, 2021
"""

# Intialize P for test example
#Left =0
#Down = 1
#Right = 2
#Up= 3

import numpy as np
import gym
from gym import wrappers
from scipy import linalg as la


P = {s : {a: [] for a in range(4)} for s in range(4)}
P[0][0] = [(0, 0, 0, False)]
P[0][1] = [(1, 2, -1, False)]
P[0][2] = [(1, 1, 0, False)]
P[0][3] = [(0, 0, 0, False)]
P[1][0] = [(1, 0, -1, False)]
P[1][1] = [(1, 3, 1, True)]
P[1][2] = [(0, 0, 0, False)]
P[1][3] = [(0, 0, 0, False)]
P[2][0] = [(0, 0, 0, False)]
P[2][1] = [(0, 0, 0, False)]
P[2][2] = [(1, 3, 1, True)]
P[2][3] = [(1, 0, 0, False)]
P[3][0] = [(0, 0, 0, True)]
P[3][1] = [(0, 0, 0, True)]
P[3][2] = [(0, 0, 0, True)]
P[3][3] = [(0, 0, 0, True)]


# Problem 1
def value_iteration(P, nS, nA, beta = 1, tol=1e-8, maxiter=3000):
    """Perform Value Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
       v (ndarray): The discrete values for the true value function.
       n (int): number of iterations
    """
    #initialize v0
    v0 = np.zeros(nS)
    v1 = np.zeros(nS)
    #iterate through max iterations
    for i in range(maxiter):
        #iterate through states
        for s in range(nS):
            #set temporary vector
            sa_vector = np.zeros(nA)
            #iterate through actions
            for a in range(nA):
                for tuple_info in P[s][a]:
                    p, s_, u, _ = tuple_info
                    #compute porbablity
                    sa_vector[a] += (p * (u + beta * v0[s_]))

            #get max value
            v1[s] = np.max(sa_vector)

        #check convergence
        if la.norm(v0 - v1, ord=2) < tol:
            break

        #update
        v0 = v1.copy()

    return v1, i + 1


# Problem 2
def extract_policy(P, nS, nA, v, beta = 1.0):
    """Returns the optimal policy vector for value function v

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        v (ndarray): The value function values.
        beta (float): The discount rate (between 0 and 1).

    Returns:
        policy (ndarray): which direction to move in from each square.
    """

    pi = np.zeros(nS)
    #we now extract the policy
    for s in range(nS):
        #create temp vector
        temp = np.sum([[p * (u + beta * v[s_]) for p, s_, u, _ in P[s][a]]
                                   for a in range(nA)], axis=1)
        #set index corresponding to max value as pi
        pi[s] = np.argmax(temp)

    return pi

# Problem 3
def compute_policy_v(P, nS, nA, policy, beta=1.0, tol=1e-8):
    """Computes the value function for a policy using policy evaluation.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        policy (ndarray): The policy to estimate the value function.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.

    Returns:
        v (ndarray): The discrete values for the true value function.
    """
    #initiilzise vectors
    v0 = np.ones(nS)
    v1 = np.zeros(nS)

    iterate = True
    while iterate:
        #iterate through states
        for s in range(nS):
            #set temporary vector
            temp = np.zeros(nA)
            #iterate through policy
            for i, pol in enumerate(policy):
                #iterate through the markov relationships
                for tuple_info in P[s][pol]:
                    p, s_, u, _ = tuple_info
                    #compute probability
                    temp[i] += (p * (u + beta * v0[s_]))

            #set element
            v1[s] = np.max(temp)

        #check convergence
        if la.norm(v0-v1, ord=2) < tol:
            break

        #update
        v0 = v1.copy()

    return v1

# Problem 4
def policy_iteration(P, nS, nA, beta=1, tol=1e-8, maxiter=200):
    """Perform Policy Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
    	v (ndarray): The discrete values for the true value function
        policy (ndarray): which direction to move in each square.
        n (int): number of iterations
    """

    #intialize pi
    pi0 = np.zeros(nS)

    #iterate until maxiter
    for k in range(maxiter):
        #compute new value vector
        v1 = compute_policy_v(P, nS, nA, pi0, beta=beta, tol=tol)
        #extract policy
        pi1 = extract_policy(P, nS, nA, v1, beta=beta)

        #check convergence
        if la.norm(pi0 - pi1, ord=2) < tol:
            break

        pi0 = pi1.copy()

    return v1, pi1, k+1




# Problem 5 and 6
def frozen_lake(basic_case=True, M=1000, render=False):
    """ Finds the optimal policy to solve the FrozenLake problem

    Parameters:
    basic_case (boolean): True for 4x4 and False for 8x8 environemtns.
    M (int): The number of times to run the simulation using problem 6.
    render (boolean): Whether to draw the environment.

    Returns:
    vi_policy (ndarray): The optimal policy for value iteration.
    vi_total_rewards (float): The mean expected value for following the value iteration optimal policy.
    pi_value_func (ndarray): The maximum value function for the optimal policy from policy iteration.
    pi_policy (ndarray): The optimal policy for policy iteration.
    pi_total_rewards (float): The mean expected value for following the policy iteration optimal policy.
    """
    raise NotImplementedError("Problem 5 Incomplete")

# Problem 6
def run_simulation(env, policy, render=True, beta = 1.0):
    """ Evaluates policy by using it to run a simulation and calculate the reward.

    Parameters:
    env (gym environment): The gym environment.
    policy (ndarray): The policy used to simulate.
    beta float: The discount factor.
    render (boolean): Whether to draw the environment.

    Returns:
    total reward (float): Value of the total reward received under policy.
    """
    raise NotImplementedError("Problem 6 Incomplete")



if __name__ == "__main__":

    '''
def value_iteration_old(P, nS, nA, beta = 1, tol=1e-8, maxiter=3000):
    """Perform Value Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
       v (ndarray): The discrete values for the true value function.
       n (int): number of iterations
    """
    #initialize v0
    v0 = np.zeros(nS)
    #iterate through max iterations
    for i in range(maxiter):
        #copy v1
        v1 = v0.copy()
        #iterate through all states
        for s in range(nS):
            #get next iteration
            v1[s] = np.max(np.sum([[p * (u + beta * v0[s_]) for p, s_, u, _ in P[s][a]]
                                   for a in range(nA)], axis=1))
        #check convergence
        if la.norm(v0 - v1, ord=2) < tol:
            break
        #reassign v1 to be v0
        v0 = v1

    return v1, i + 1
    '''

    #question 1
    '''
    solution = np.array([1, 1, 1, 0])
    num_iters = 3
    v, i = value_iteration(P, 4 ,4, beta = 1, tol=1e-8, maxiter=3000)
    print(np.allclose(v, solution ))
    print(i == num_iters)
    '''

    #question 2
    '''
    policy = np.array([2, 1, 2, 0])
    c = extract_policy(P, 4, 4, v, beta = 1.0)
    print(np.allclose(policy, c))
    '''

    #question 3
    '''
    my_policy = compute_policy_v(P, 4, 4, policy, beta=1.0, tol=1e-8)
    print(my_policy)
    print(np.allclose(compute_policy_v(P, 4, 4, policy, beta=1.0, tol=1e-8), solution))
    '''


    #question 4
    #print(policy_iteration(P, 4, 4, beta=1, tol=1e-8, maxiter=200))
