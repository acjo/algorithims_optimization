#DFT.py

import numpy as np

from numpy.linalg import inv

def DFT(f):
    '''
       Computes the discrete Fourier transform of the 1D array f.
    '''

    n = len(f)
    m = np.arange(n).reshape(n,1)
    W = np.exp((-2j * np.pi / n) * m @ m.T) / n


    return W @ f , W


_, W = DFT(np.array([1, 1, 0, 0]))

Winv = inv(W)

print(Winv @ np.array([1, 0, 1, 0]))


