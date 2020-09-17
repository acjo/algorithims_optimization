#osman_caelan_1_8.py
"""
Caelan Osman
Math 320 Sec 1
September 16 2020
"""

#Python program for Extended Euclidean algorithm
def extended_euclidean_algorithm(a, b):
    """ This program returns a tuple with the greatest common divisor of a,b and
        x,y such that d = ax + by
    """
    if a == 0: #in the case where a is zero below will always be the case
        return b, 0, 1
    gcd, xn, yn = extended_euclidean_algorithm(b% a, a)
    #the above is a recursive call, it passes in the remainder as the lower bound and
    #the old lower bound as the upper bound

    x = yn - (b//a) * xn
    if gcd < 0: #make sure the gcd is positive
        gcd = gcd * -1
    y = xn

    return gcd, x, y

