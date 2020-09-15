# numpy_intro.py
"""Python Essentials: Intro to NumPy.
Caelan Osman
Math 321 Sec 3
September 10, 2020
"""
import numpy as np

def prob1():
    """Define the matrices A and B as arrays. Return the matrix product AB."""
    A = np.array([[3,-1, 4], [1, 5, -9]])
    B = np.array([[2, 6, -5, 3],[5,-8,9,7], [9, -3,-2, -3]])
    return A @ B

def prob2():
    """Define the matrix A as an array. Return the matrix -A^3 + 9A^2 - 15A."""
    A = np.array([[3, 1, 4], [1, 5, 9], [-5, 3, 1]])
    return -A@A@A + 9 * A@A-15 * A

def prob3():
    """Define the matrices A and B as arrays. Calculate the matrix product ABA,
    change its data type to np.int64, and return it.
    """
    A = np.triu(np.ones((7,7)))
    B = np.triu(np.full((7,7), 5)) + np.tril(np.full((7,7), -1)) + -5*np.eye(7)
    return A@B@A.astype(np.int64)

def prob4(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy."""
    cpy = np.copy(A)
    mask = cpy < 0
    cpy[mask] = 0
    return cpy

def prob5():
    """Define the matrices A, B, and C as arrays. Return the block matrix
                                | 0 A^T I |
                                | A  0  0 |,
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    A = np.arange(6).reshape((3,2)).T
    B = np.tril(np.full((3,3), 3))
    C = np.diag([-2, -2, -2])
    col_1 = np.vstack((np.zeros((3,3)), A, B))
    col_2 = np.vstack((A.T, np.zeros((2,2)), np.zeros((3,2))))
    col_3 = np.vstack((np.eye(3), np.zeros((2,3)), C))
    block = np.hstack((col_1, col_2, col_3))
    return block

def prob6(A):
    """Divide each row of 'A' by the row sum and return the resulting array.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    sums = A.sum(axis=1)
    sums = np.vstack(sums)
    return A / sums

def prob7():
    """Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid.
    """
    grid = np.load("grid.npy")
    new = grid[:,:-3]
    max_row = np.max(grid[:,:-3] * grid[:,1:-2] * grid[:,2:-1] * grid[:,3:])
    max_col = np.max(grid[:-3,:] * grid[1:-2,:] * grid[2:-1, :] * grid[3:, :])
    max_diagonal_up = np.max(grid[:-3,:-3] * grid[1:-2, 1:-2] * grid[2:-1, 2:-1] * grid[3:, 3:] )
    max_diagonal_down = np.max(grid[3:,:-3] * grid[2:-1, 1:-2] * grid[1:-2, 2:-1] * grid[:-3, 3:] )
    return np.max([max_row, max_col, max_diagonal_up, max_diagonal_down])
