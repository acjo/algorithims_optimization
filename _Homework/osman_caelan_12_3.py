#osman_caelan_12_3.py

import numpy as np
from scipy import linalg as la

def exact_gradient_descent_quadratic(eps,x0,b,Q):


    while la.norm(Q @ x0 - b, ord=2) >= eps:
        #calculate alpha
        alpha = np.inner(Q @ x0 - b, Q @ x0 - b) / ((Q @ x0 - b).T @ Q @ (Q @ x0 - b))
        #calculate next point
        x0 = x0 - alpha*(Q @ x0 - b)

    return x0



if __name__ == "__main__":

    A = np.random.random((3, 3))

    Q = A + A.T

    print(exact_gradient_descent_quadratic(1e-5, np.array([1, 2, 3]), np.array([3, 5, 6]), Q))