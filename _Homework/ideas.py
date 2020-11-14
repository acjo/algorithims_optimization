import numpy as np

A = np.array([[1.1, -0.5, 1, 0],
              [3, 1/4., 1/5., 1/6.],
              [1/3., 1/10, 1/20., 1/16.],
              [-3, -0.11, 8, 0]])
print(A)

print()
print(np.where(np.where(A > 1, 1, A) < 0, 0, np.where(A > 1, 1, A)))
print()
print(np.clip(A, 0, 1))
