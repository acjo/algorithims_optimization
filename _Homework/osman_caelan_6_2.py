#osman_caelan_6_2.py

import numpy as np
from scipy.stats import bernoulli as bigb

def bernoulli(p):
    '''
    This function samples from the beta distribution and compares the mean estimate with the actual and the proportionality of the difference
    '''
    #contains epsilon values
    eps = [0.1, 0.01, 0.001]
    #contains proportions
    proportions = []
    #contains upperBounds
    upperBounds = []
    #sample size
    n = 1000
    #expected value
    E = p
    #variance given by the book
    sigma_2 = p * (1 - p)

    #repeat for each epislon value
    for epsilon in eps:
        #calculates the upper bound
        bound = sigma_2 / (n * (epsilon ** 2))
        upperBounds.append(bound)

        #calculates the difference between our estimated mean and the actual
        differences = np.array([abs(sum(np.random.binomial(1, p, n)) / n - E) for k in range(0, 100)])
        #creates a mask for which values are greater than epislon
        mask = differences >= epsilon
        #calculates the proportionality
        proportion = len(differences[mask]) / 100
        proportions.append(proportion)

    return np.array(proportions), np.array(upperBounds)


def beta(a,b):
    '''
    This function samples from the beta distribution and compares the mean estimate with the actual and the proportionality of the difference
    '''
    #contains all epsilons
    eps = [0.1, 0.01, 0.001]
    #initializes upperBuonds and proportions
    proportions = []
    upperBounds = []
    #number of sample times
    n = 1000
    #expected value
    E = a/(a+b)
    #variance
    sigma_2 = (a*b)/(((a+b)**2) * (a+b+1))


    #repeat for each epsilon value
    for epsilon in eps:
        #calculate upper bound
        bound = sigma_2 / (n * (epsilon ** 2))
        upperBounds.append(bound)

        #gets the differences in our estimated mean and the actual
        differences = np.array([abs(sum(np.random.beta(a, b, n)) / n - E) for k in range(0, 100)])
        #creates a mask for which the values are greater than epsilon
        mask = differences >= epsilon
        #calculates the current proportionality
        proportion = len(differences[mask]) / 100
        proportions.append(proportion)

    return np.array(proportions), np.array(upperBounds)
