#test_change_of_basis.py

import pytest
import numpy as np
from matplotlib import pyplot as plt
import bernstein_functions as bf
import change_of_basis as cob

def test_fft():
    x_4 = np.zeros(5)
    x_4[-1] = 1
    exact = np.array([3/8., 0, 1/2., 0, 1/8.])
    assert np.allclose(cob.bernsteinToChebyshev(x_4), exact)

def test_transition():
    factor = np.sqrt(np.pi) / np.sqrt(2)
    cM1 = factor * np.array([[1/2, 1/2],
                             [-np.sqrt(2)/4, np.sqrt(2)/4]])

    cM2 = factor * np.array([[3/8., 1/4., 3/8.],
                             [-np.sqrt(2)/4., 0, np.sqrt(2)/4.],
                             [np.sqrt(2)/16., -np.sqrt(2)/8., np.sqrt(2)/16.]])

    cM3 = factor * np.array([[5/16., 3/16., 3/16., 5/16.],
                             [-15*np.sqrt(2)/64, -3*np.sqrt(2)/64, 3*np.sqrt(2)/64, 15*np.sqrt(2)/64],
                             [3*np.sqrt(2)/32, -3*np.sqrt(2)/32, -3*np.sqrt(2)/32, 3*np.sqrt(2)/32],
                             [-np.sqrt(2)/64, 3*np.sqrt(2)/64, -3*np.sqrt(2)/64,np.sqrt(2)/64]])

    assert np.allclose(cM1, cob.transitionMatrixBTC( 1 ))
    assert np.allclose(cM2, cob.transitionMatrixBTC( 2 ))
    assert np.allclose(cM3, cob.transitionMatrixBTC( 3 ))


print(cob.transitionMatrixBTC( 1 ))

x4 = np.zeros(5)
x4[-1] = 1
chebyFFT = cob.bernsteinToChebyshev(x4)
#test for x^4 using matrix
chebyMatrix = cob.bernsteinToChebyshev(x4, matrix=True)

chebyBasis = [bf.chebyshevFunc(j) for j in range(5)]
chebyBasis01 = [bf.chebyshevFunc(j, domain=[0, 1]) for j in range(5)]


original = lambda x: x**4

#function from fft
x4FFT = lambda x : np.sum([chebyFFT[i] * f(x) for i, f in enumerate(chebyBasis)])
x4Matrix = lambda x : np.sum([chebyMatrix[i] * f(x) for i, f in enumerate(chebyBasis01)])

ax = plt.axes()
ax.set_facecolor('gray')


domain1 = np.linspace(-1, 1, 100)
domain2 = np.linspace(0, 1, 100)

outputFFT = np.array([x4FFT(x) for x in domain1])
outputMatrix = np.array([x4Matrix(x) for x in domain2])
plt.plot(domain1, original(domain1), 'w', linewidth=7, label='original')
plt.plot(domain1, outputFFT, 'b', linewidth=4, label='COB using FFT')
plt.plot(domain2, outputMatrix, 'r', linewidth=1, label='COB using Matrix')
plt.legend(loc='best')
plt.title(r'$x^4$ in different polynomial bases')
plt.show()
