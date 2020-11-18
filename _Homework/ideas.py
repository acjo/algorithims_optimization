import numpy as np
from numpy import linalg as la
from scipy import linalg as spla

A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])

x = np.array([1, 5, 9, 13])


print(spla.norm(A - np.vstack(x), axis = 0))
