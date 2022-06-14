#periodic_sampling_thm.py

import sys
from typing import Union, Any, Callable
import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft


#helper function, adds two lambda functions
def function_add(f1:Callable[[], Any], f2:Callable[[], Any]):
    """Adds two lambda functions together.
    Paramaters:
        f1 (callable) first function
        f2 (callable) second function
    Returns:
        f1 + f2 (callable): new callable that adds together the functions 
    """
    return lambda *args, **kwds: f1(*args, **kwds) + f2(*args, **kwds)

#helper function perfoms DFT
def DFT(f:Union[np.ndarray, list]):
    """Computes the discrete Fourier Transform of the 1d array f.
    Paramters:
        f (np.ndarray, list): array containing values to perform the DFT on.
    Returns:
        DFT(f) (np.ndarray): The DFT of f. 
    """
    n = len(f)
    m = np.arange(n).reshape(n, 1)
    W = np.exp((-2j*np.pi/n)*m @ m.T)
    return W @ f / n

#helper function performs FFT
def FFT(f):
    """Computes the fast Fourier transform. 
    Note that this function assumes that the length of f is a power of 2.
    Paramters:
        f (np.ndarray, list): array containing values to perform the DFT on. 
    Returns:
        FFT(f) (np.ndarray): The FFT of f. 
    """
    n = len(f)

    if n <= 4:
        return DFT(f)
    else:
        f_even = FFT(f[::2])
        f_odd = FFT(f[1::2])
        w = np.exp((-2j*np.pi/n)*np.arange(n))
        first_sum = f_even + w[:n//2]*f_odd
        second_sum = f_even + w [n//2:]*f_odd
        return 0.5 * np.concatenate([first_sum, second_sum])

def periodic_sampling_theorem(f:Union[Callable[[], Any], np.ndarray, list], T:float, n:int, nu:int, call:bool=False):
    """Uses the preidoic sampling theorem to recreate a function f
    with n sample points an Nyquist freqency nu. 

    Parameters:
        nu (int): the Nyquist frequency
        n (int): the number of samples.
        T (float): the domain of the function
        f (callable, vector): the function to recreate or a sample 
        call (bool): if True, return a callable function.

    Returns:
        coeff (np.ndarray): the coefficients given by the Periodic Sampling Theorem
        pst (callable): if call is True, a callable function that recreates the representative function 
        from the samples using the periodic sampling theorem. 
    
    Raises:
        ValueError: if the number of sampling points n does not exceed the Nyquist rate 2*nu
    """

    if n <= 2*nu:
        raise ValueError("The number of samples {} is not enough samples to gurantee uniqueness with the Nyquist frequency {}.".format(n, nu))

    omega = 2 * np.pi / T
    domain = np.arange(0, T, T / n) #get sample points

    if isinstance(f, Callable):
        fhat = fft(f(domain)) / n
    else:
        fhat = fft(f) / n

    #compute coefficients
    coeff = np.array([ fhat[k+n] if k < 0 else fhat[k] for k in range(-nu, nu+1)])

    # create callable
    if call:
        #compute pst
        def pst(t) :
            return np.real(np.sum([coeff[k + nu]*np.exp(1j*omega*k*t) for k in range(-nu, nu + 1)], axis=0))
        return pst, coeff

    #otherwise return the coefficients
    return coeff

def antialiasing():

    return


def main(key):

    if key == "1":
        y = lambda x: np.sin(4*x)

        pst, _ = periodic_sampling_theorem(y, 2*np.pi, 3, 1, call=True)

        domain = np.linspace(0, 2*np.pi, 1000)

        fig = plt.figure()
        fig.set_dpi(150)
        ax = fig.add_subplot(111)
        ax.plot(domain, pst(domain), "r-", linewidth=2, label="Periodic Sampling Thm")
        ax.plot(domain, y(domain), "b--", markersize=1, label="Original function")
        ax.set_title(r"$f(x) = \sin(x)$")
        ax.legend(loc="best")
        plt.show()

        def approximate(n, f):
            T = 1
            nu = n // 2
            domain = np.arange(0, T, T/n)
            sample = f(domain)

            return sample, *periodic_sampling_theorem(f, T, n, nu, call=True)

        f = lambda x: 1 - 3 * np.sin( 12 * np.pi * x + 7 ) + 5 * np.sin( 2 * np.pi * x - 1 ) + 5 * np.sin( 4 * np.pi * x - 3 )
        domain = np.linspace( 0, 1, 200 )

        n = [3, 7, 11, 13]

        fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(10, 5))
        fig.set_dpi(150)

        i = 0
        for j in range( 2 ):
            for k in range( 2 ):
                ax = axs[j, k]
                points = np.arange(0, 1, 1 / n[i])
                sample, pst, coeff = approximate(n[i], f)
                ax.plot(points, sample, "k.", markersize=2, label="samples")
                ax.plot(domain, f(domain), 'b-', linewidth=3, label = "original")
                ax.plot(domain, pst(domain),'r-', linewidth=1, label = "pst")
                ax.set_title(r"$n = {{{}}}$, $\nu = {{{}}}$".format(n[i], n[i]//2))
                ax.legend(loc="best")
                i += 1

        plt.suptitle(r"$f(x) = 1-3\sin(12\pi x + 7) + 5\sin(2\pi x-1) + 5\sin(4\pi x -3 )$")
        plt.show()

    elif key=="2":
        pass

    else:
        raise ValueError ("{} is an incorrect problem specification.".format(key))

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
