import numpy as np
from numpy import linalg as la
from scipy import linalg as spla

def compact_svd(A, tol=1e-6):
    eigs, V = la.eig(A.conj().T @ A)
    singular_vals = np.sqrt(abs(eigs))
    idx = np.argsort(singular_vals)[::-1]
    sigma = singular_vals[idx]
    V1 = V[idx]


    return sigma, V1.conj().T


A = np.random.random((8,16))

u,s, vh = la.svd(A, full_matrices=False)

s1, v1 = compact_svd(A)

print(s)

print(s1)
