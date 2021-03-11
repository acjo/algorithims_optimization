# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
Caelan Osman
Math 323 Sec. 1
March 10, 2021
"""

import cvxpy as cp
import numpy as np

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #creat variable
    x = cp.Variable(3, nonneg=True)

    #coefficient array
    c = np.array([2, 1, 3])

    #objective function
    objective = cp.Minimize(c.T @ x)

    #All constraints
    A = np.array([1, 2, 0])
    B = np.array([0, 1, -4])
    C = np.array([2, 10, 3])
    P = np.eye(3)
    constraints = [A @ x <=3, B @ x <=1, C @ x >= 12]

    #create, solve, and return solution
    problem = cp.Problem(objective, constraints)
    solution = problem.solve()
    return x.value, solution


# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #create variable
    x = cp.Variable(A.shape[1], nonneg=True)

    #create objective and constraints
    objective = cp.Minimize(cp.norm(x, 1))
    constraints = [A @ x == b]

    #solve and return solution and optimizer
    problem = cp.Problem(objective, constraints)
    solution = problem.solve()

    return x.value, solution


# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #number of pianos along each route
    p = cp.Variable(6, nonneg=True)
    #cost of transportation
    c = np.array([4, 7, 6, 8, 8, 9])

    objective = cp.Minimize(c.T @ p)

    ones = np.ones(6)

    constraints = [ones @ p == 7, ones @ p == 2, ones @ p ==4, ones @ p <= 5, ones @ p <= 8]
    problem = cp.Problem(objective, constraints)
    solution = problem.solve()

    return p.value, solution


# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file
    food.npy to create a convex optimization problem. The first column is
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal
    objective.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    raise NotImplementedError("Problem 6 Incomplete")

if __name__ == "__main__":

    #problem 1
    #print(prob1())

    #problem 2
    '''
    A = np.array([[1, 2, 1, 1], [0, 3, -2, -1]])
    b = np.array([7, 4])
    print(l1Min(A, b))
    '''

    #problem 3
    #print(prob3())
