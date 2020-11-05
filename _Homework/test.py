import numpy as np
import random as rand

A = np.array([[1, 0.5, 1/3.], [0, 0, 1/3.], [0., 0.5, 1/3.]])

print(np.allclose(A.sum(axis=0), np.ones(A.shape[1])))

