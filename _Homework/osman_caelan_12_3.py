#osman_caelan_12_3.py

import numpy as np
from scipy import linalg as la

def exact_gradient_descent_quadratic(eps, x0, b, Q):
    #initialize variable and functions
    df = lambda x: Q @ x - b
    alpha_k = lambda x: (df(x).T @ df(x)) / (df(x).T @ Q @ df(x))

    #x = x0

    #perform descent until we meet the stopping criterion
    while la.norm(df(x0).T) >= eps:
        x0 = x0 - alpha_k(x0)*df(x0)

    return x0

'''
if __name__ == "__main__":
    Q = np.array([[6,-9],[-9,21]])
    b = np.array([[10],[-26]])
    x_0 = np.array([[.27],[.74]])

    print(exact_gradient_descent_quadratic(1e-6, x_0, b, Q))
'''
