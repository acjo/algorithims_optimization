import numpy as np
from numpy.fft import fft
from numpy.polynomial.chebyshev import Chebyshev as Cheb
from matplotlib import pyplot as plt


def cheb_interp(n):
    ''' Compute the coefficients of the degree-n Chebyshev
    interpolation of f at the extremizers y_k = cos(k pi/ n)
    '''

    y = np.cos((np.pi * np.arange(2*n)) / n)
    evals = np.piecewise(y, [y < 0, y >= 0], [lambda y: y+1, lambda y: y])
    coeffs = np.real(fft(evals))[:n+1] / n

    coeffs[0] /= 2
    coeffs[n] /= 2

    return coeffs


k = [i for i in range(1, 9)]
fig, axs = plt.subplots(2, 4)
domain = np.linspace(-1, 1, 200)

a = 0
for i in range(2):
    for j in range(4):
        ax = axs[i, j]
        if a == 7:
            output = np.piecewise(domain, [domain < 0, domain >= 0],
                                  [lambda domain: domain+1, lambda domain: domain])
            ax.plot(domain, output, label='f(x)')
            ax.legend(loc='best')

        else:
            n = 2**k[a]
            current_coeffs = cheb_interp(n)
            current_cheby = Cheb(current_coeffs)
            output = current_cheby(domain)
            ax.plot(domain, output, label ='n = ' + str(n))
            output = np.piecewise(domain, [domain < 0, domain >= 0],
                                  [lambda domain: domain+1, lambda domain: domain])
            ax.plot(domain, output, label='f(x)')
            ax.legend(loc='best')
        a += 1

plt.show()
