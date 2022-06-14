#haar_son_approx.py

from ast import Call
import sys
from typing import Union, Any, Callable
import numpy as np
from matplotlib import pyplot as plt

def sampling_points(f:Callable[[], Any], n:int):
    """takes in a function and a positive integer n and 
    returns the wavelet sampling points {k/2**n} where 0 < k < 2**n - 1.
    Parameters:
        f (callable): function to sample at.
        n (int): integer factor to get sampling points.
    Returns: 
        points (np.ndarray): the sampling points.
        f(points) (np.ndarray): the samples.
    Raises:
        ValueError:if n is nonpostive.
    """

    if n <= 0:
        raise ValueError("{} is a nonpositive integer.".format(n))

    points = np.array([k/(2**n) for k in range(2**n)])

    return points, f(points)

def haar_father(x):
    """Function evaluating the Haar father wavelet.
    Parameters:
        x (float, int, np.ndarray, list): evaluation points.

    Returns:
        f(x) (float, np.ndarray): the evaluation of the haar 
        father function.
    """
    return np.where((0 <= x) & (x < 1), 1., 0.)

def haar_sons_approximation(f:Callable[[], Any], n:int, x):
    """Creates and evaluates an approximation of the function f
    using a linear combination of Haar son wavelets. Remember 
    given that f is left continuous with compact support, the 
    haar son approximation of f converges pointwise.
    Parameters:
        f (callable): function to approximate
        n (int): integer factor to get sampling points
        x (int, float, np.ndarray, list): the evaluation point(s).

    Returns:
        (float, np.ndarray): array of points the evaluated approximation function.
    
    Raises:
        ValueError: if n is nonpositive.
    """

    def _sampling(f:Callable[[], Any], n:int):
        """takes in a function and a positive integer n and 
        returns the wavelet sampling points {k/2**n} where 0 < k < 2**n - 1.

        Parameters:
            f (callable): function to sample at.
            n (int): integer factor to get sampling points.
        Returns: 
            f(points) (np.ndarray): the samples.
        Raises:
            ValueError: if n is nonpostive.
        """

        points = np.array([k/(2**n) for k in range(2**n)])

        return f(points)

    if n <= 0:
        raise ValueError("{} is a nonpositive integer.".format(n))

    # get sampling points
    a = _sampling(f, n)

    # return the approximation of f using wavelets. 
    return np.sum([np.where(((k/(2**n)) <= x) & (x < (k + 1)/(2**n)), coeff, 0) for k, coeff in enumerate(a)], axis=0) 

def main(key:str):
    """Test for this module
    """

    if key == "test":
        # function to approximate
        def func(t):
            return np.sin((2*np.pi*t) - 5)/np.sqrt(abs(t - (np.pi/20)))

        # 2**n evaluation points
        n = np.arange(1, 11)

        # set up figure to evaluate approximation
        fig, axs = plt.subplots(2, 5, constrained_layout=True, figsize=(10, 10))
        fig.set_dpi(200)
        domain = np.linspace(0, 1, 1000, endpoint=False)
        i = 0
        for j in range(2):
            for k in range(5):
                ax = axs[j, k]
                # plot original function
                ax.plot(domain, func(domain), "b-", linewidth=3, label="Exact Function")
                # plot approximation
                ax.plot(domain, haar_sons_approximation(func, n[i], domain), "r--", linewidth=1.5, label="Wavelet Approximation.")
                ax.set_title("# of Sampling Points: {}".format(2**n[i]))
                ax.legend(loc="best")
                i += 1

        plt.suptitle(r"Wavelet Approximation of $f(x) = \frac{\sin(2\pi t -5)}{\sqrt{|t - \pi/20|}}$")
        plt.show()

    else:
        raise ValueError ("{} is an incorrect problem specification.".format(key))

    return

if __name__ == "__main__":

    if len(sys.argv) == 2:
        main(sys.argv[1])
