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

import sys
import numpy as np
import gym
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
                    #compute probablity
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

    pi = np.zeros(nS, dtype=int)
    #we now extract the policy
    for s in range(nS):
        #create temp vector
        sa_vector = np.zeros(nA)
        #populate the temporary vector
        for a in range(nA):
            for next_sr in P[s][a]:
                p, s_, u, _ = next_sr
                sa_vector[a] += (p * (u + beta * v[s_]))

        #set index corresponding to max value as pi
        pi[s] = np.argmax(sa_vector)

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
    v0 = np.zeros(nS)

    while True:
        #iterate through states
        v1 = v0.copy()
        for s in range(nS):
            policy_a = policy[s]
            #set temporary vector
            v0[s] = sum([p * (u + beta * v1[s_]) for p, s_, u, _ in P[s][policy_a]])

        #check convergence
        if (np.sum((np.fabs(v0 - v1))) <= tol):
            break

    return v0

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
    policy = np.random.choice(nA, size=nS)

    #iterate until maxiter
    for k in range(maxiter):
        # compute value function
        old_policy_v = compute_policy_v(P, nS, nA, policy, beta=beta)
        # extract polciy from value function
        new_policy = extract_policy(P, nS, nA, old_policy_v, beta=beta)

        #check convergence
        if np.all(policy == new_policy):
            break

        policy = new_policy.copy()

    return old_policy_v, policy, k + 1

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
    if basic_case:
        env_name  = 'FrozenLake-v1'
    else:
        env_name  = 'FrozenLake8x8-v1'

    env = gym.make(env_name).env

    nS = env.observation_space.n
    nA = env.action_space.n
    vi_value_func, vi_iterations = value_iteration(env.P, nS, nA)
    vi_policy = extract_policy(env.P, nS, nA, vi_value_func,)
    pi_value_func, pi_policy, pi_iterations = policy_iteration(env.P, nS, nA)

    # Problem 6
    vi_total_rewards = np.mean([run_simulation(env, vi_policy, render) for _ in range(M)])
    pi_total_rewards = np.mean([run_simulation(env, pi_policy, render) for _ in range(M)])

    return vi_policy, vi_total_rewards, pi_value_func, pi_policy, pi_total_rewards

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
    obs = env.reset()
    total_reward = 0
    step_index = 0
    while True:
        if render == True:
            env.render(mode='human')
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (beta ** step_index * reward)
        step_index += 1
        if done:
            break
    return total_reward


def main(key):

    if key == "1":
        solution = np.array([1, 1, 1, 0])
        num_iters = 3
        v, i = value_iteration(P, 4 ,4, beta = 1, tol=1e-8, maxiter=3000)
        assert np.allclose(v, solution)
        assert i == num_iters

    elif key == "2":
        v, i = value_iteration(P, 4 ,4, beta = 1, tol=1e-8, maxiter=3000)
        policy = np.array([2, 1, 2, 0])
        c = extract_policy(P, 4, 4, v, beta = 1.0)
        assert np.allclose(policy, c)

    elif key == "3":
        value_function_iter, _ = value_iteration(P, 4 ,4, beta = 1, tol=1e-8, maxiter=3000)
        policy = np.array([2, 1, 2, 0])
        value_function_poicy = compute_policy_v(P, 4, 4, policy, beta=1.0, tol=1e-8)
        assert np.allclose(value_function_iter, value_function_poicy)

    elif key == "4":
        value_func, policy, _ = policy_iteration(P, 4, 4, beta=1, tol=1e-8, maxiter=200)
        assert np.allclose(policy, np.array([2, 1, 2, 0]))
        assert np.allclose(value_func, np.array([1, 1, 1, 0]))

    elif key == "5":
        vi_policy, vi_total_rewards, pi_value_func, pi_policy, pi_total_rewards = frozen_lake(basic_case=True, M=1000, render=False)
        value_func = np.array([0.82352937, 0.82352936, 0.82352935, 0.82352934, 0.82352937, 
        0., 0.52941174, 0., 0.82352938, 0.82352939, 0.76470586, 0., 0., 0.88235292, 0.94117646, 0.])
        assert np.allclose(vi_policy, pi_policy)
        assert np.abs(vi_total_rewards - pi_total_rewards) < 0.03
        assert np.allclose(pi_value_func, value_func)

    elif key == "all":
        main("1")
        main("2")
        main("3")
        main("4")
        main("5")

    else:
        raise ValueError ("{} is an incorrect problem specification.".format(key))

    return

if __name__ == "__main__":

    if len(sys.argv) == 2:
        main(sys.argv[1])