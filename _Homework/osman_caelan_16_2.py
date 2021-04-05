#osman_caelan_16_2.py
'''
Caelan Osman
Math 322 Sec. 1
April 4, 2021
'''


import numpy as np

def optimal_value_function(T, u={1:-1, 2: 0.7, 3:-1, 4:-1, 5:-1, 6:-1, 7:-1, 8:-1, 9: -1}, mTFind = True):

    #dictionary of actions to take square: list of squares you can move to

    def value_function(T, u):
        V = {i:0 for i in np.arange(1, 10)}
        new_V = {}
        A_s = [[2, 4], [1, 3, 5], [2, 6], [1, 5, 7], [2, 4, 6, 8], [3, 5, 9], [4, 8], [5, 7, 9]]
        for t in range(T+1):
            for j, state in enumerate(A_s):
                L = [u[i] + V[i] for i in state]
                new_V[j+1] = round(max(L), 1)

            new_V[9] = 0
            V = new_V.copy()
        return V

    if mTFind:
        mT = {1: 19, 2: 18, 3: 17, 4: 18, 5: 17, 6: 6, 7: 7, 8: 6, 9: 0}
        print(mT)

    return value_function(T, u)

#antoher method
'''
    actions = {1: [2, 4], 2:[1, 3, 5], 3: [2, 6], 4:[1, 5, 7],
               5: [2, 4, 6, 8], 6:[3, 5, 9], 7: [4, 8], 8:[5, 7, 9], 9:[]}
    states = np.arange(1, 10)
    def value_function(s_val, iters):
        A_s = actions[s_val]
        if A_s == []:
            return 0
        if iters == 0:
            return max([u[action] for action in A_s])
        else:
            return max([u[a] + value_function(a, iters-1) for a in A_s])

    if mTFind:
        mT = {1: 19, 2: 18, 3: 17, 4: 18, 5: 17, 6: 6, 7: 7, 8: 6, 9: 0}
        print(mT)

    return {s: value_function(s, T) for s in states}


#print(optimal_value_function(19))
'''
