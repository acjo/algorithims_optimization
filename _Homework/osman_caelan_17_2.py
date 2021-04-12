#osman_caelan_17_2.py

import numpy as np
from numpy.random import uniform

def pull(prob, payouts, action):
    '''Pulls a multi-armed bandit machine:
       Paramaters:
           prob ((n, ) ndarray): array of success probabilities for each arm
           payouts ((n, ) ndarray): array of payouts for each arm
           action (int): the action to take in [1, ..., n]
       Returns:
           (float): amount won
           ((2, ) ndarray): the increase of success/failure for action arm
    '''
    #get the index in python
    adjust_index = action-1

    #get the corresponding probability
    action_prob = prob[adjust_index]

    #get uniform distribution draw
    draw = uniform()

    #if draw was successful return corresponding payout and change
    if draw <= action_prob:
        return payouts[adjust_index], np.array([1, 0])

    #if draw was unsuccesful return 0 as payout and change
    else:
        return 0, np.array([0, 1])

def compute_R(M, r, beta):
    ''' Computes max expected return
        Paramaters:
            M (int): Max value of a + b = M
            r (float): true expected value
            beta (float): discount factor
        Returns:
            R_values ((M+1, M+1), ndarray) where R[a, b] = R(a, b, r)
    '''
    #intiialize array
    R_values = np.zeros((M+1, M+1))

    for i in range(M, -1, -1):
        #get list of values that add up to the current i value
        values = [(a, b) for a, b in enumerate(range(i, -1, -1))]
        #if we are beginning use equation (17.10)
        if i == M:
            for a, b in values:
                R_values[a, b] = (1/(1-beta)) * max(a/(a+b), r)

        #otherwise use equation (17.7)
        else:
            for a, b in values:
                if a == 0 and b == 0:
                    continue
                R_values[a, b] = max((a * (1 + beta * R_values[a+1, b]) + b *
                                      beta * R_values[a, b+1]) / (a+b),
                                     r / (1 - beta))

    return R_values
