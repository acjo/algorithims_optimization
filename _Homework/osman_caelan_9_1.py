#osman_caelan_9_1.py

from scipy.special import binom
from matplotlib import pyplot as plt
import numpy as np


def bernstein_function(k, n, x):
    return binom(n, k) * ((x) ** k) * ((1-x)**(n-k))

def bernstein_transform(n, f, x):
    return sum([f(k/n) * bernstein_function(k, n, x) for k in range(0, n+1)])

'''
if __name__ == "__main__":
    fig, axs = plt.subplots(2, 2)
    func = lambda x: x**2 * np.sin(2 * np.pi * x + np.pi)

    domain = np.linspace(0, 1, 500)
    rnge1 = func(domain)

    ns = [4, 10, 50, 200]
    rnge2 = np.array([bernstein_transform(n, func, domain) for n in ns])

    for n in ns:
        current = np.array([bernstein_transform(n, func, val) for val in domain])
        rnge2.append(current)

    l = 0
    for i in range(2):
        for j in range(2):
            ax = axs[i, j]
            ax.plot(domain, rnge1, 'r')
            ax.plot(domain, rnge2[l], 'k')
            ax.set_title('n = ' + str(ns[l]))
            l += 1


    plt.show()
'''
