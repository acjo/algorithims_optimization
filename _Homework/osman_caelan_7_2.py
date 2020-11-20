#osman_caelan_7_2.py

import numpy as np
from scipy.stats import norm
from scipy.stats import beta
from scipy import linalg as la
import time


def exercise7_6(part):

    #the function we want to integrate
    func = lambda x: np.exp(-x**2 / 2)
    #total number of samples
    n = 10**5

    #where to take the sample from
    if part == 1:
        #get the sample
        sample = np.random.normal(0, 1, n)
        #apply indicator indicator RV. Divide functioin value by pdf falue if the sample is greater than 3, 0 otherwise
        values = [func(sample[i]) / norm.pdf(sample[i]) if sample[i] >= 3 else 0 for i in range(n)]
    elif part == 2:
        sample = np.random.normal(3, 1, n)
        values = [func(sample[i]) / norm.pdf(sample[i], loc=3) if sample[i] >= 3 else 0 for i in range(n)]
    else:
        raise ValueError('Not Implimented')

    #sum up values and divide by the total amount to get estimated integral
    estimate = (1 / n) * sum(values)
    #use unbiased sample variance
    variance = (1 / (n - 1)) * sum((values - estimate)**2)
    #estimate standard error
    std_err = np.sqrt(variance / n)

    return estimate, std_err


def exercise7_7():
    #function we want to integrate
    func = lambda x: 1/ (x**3 + x + 1)

    #change of coordinates function
    coc = lambda x: 2*np.pi*x

    #composition of function and coc
    comp = lambda x: func(coc(x))

    a, b, n = 1, 5, 10**6

    sample = np.random.beta(a, b, n)

    values = [(np.pi * 2 * comp(sample[i]))/ (beta.pdf(sample[i], a, b)) if sample[i] >= 0 and sample[i] <= 1 else 0 for i in range(n)]

    estimate = (1/n) * sum(values)

    variance = (1/(n-1)) * sum((values - estimate)**2)

    std_err = np.sqrt(variance / n)


    return a, b, n, std_err

def dDimensionalUnitBall(d):
    n = 5 * 10**7
    points = np.random.uniform(-1, 1, (d, n))
    lengths = la.norm(points, axis=0)
    num_within = np.count_nonzero(lengths < 1)

    return 2**d * (num_within / n)
