#osman_caelan_7_1.py

import numpy as np


def estimate_pi(n):

    num = 10**n

    #get the guesses for X and Y
    x = np.random.uniform(-1, 1, num)
    y = np.random.uniform(-1, 1, num)

    radius = np.sqrt(x**2 + y**2)

    #get the number of interior points also the x
    #will now represent 1 for inside or no for outside
    interior_count = 0
    for i in range(10**n):
        if radius[i] <= 1:
            interior_count += 1
            x[i] = 1
        else:
            x[i] = 0

    pi = 4 * interior_count / num

    #expected value is the number of 1's (the interior
    # count) divided by the total number of points
    mean = (interior_count / num)

    variance = (1/(num - 1)) * sum((x - mean)**2)

    std_err = np.sqrt(variance / num)

    return pi, std_err

