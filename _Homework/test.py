import numpy as np
import random as rand



u = np.random.randint(0, 10, (3, 1))
v = np.random.randint(0, 10, (3, 1))
print(v @ u.T)
