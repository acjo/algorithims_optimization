# specs.py
"""Python Essentials: Unit Testing.
Caelan Osman
Math 321 sec 3
September 27, 2020
"""

from itertools import combinations

def add(a, b):
    """Add two numbers."""
    return a + b

def divide(a, b):
    """Divide two numbers, raising an error if the second number is zero."""
    if b == 0:
        raise ZeroDivisionError("second input cannot be zero")
    return a / b


# Problem 1
def smallest_factor(n):
    """Return the smallest prime factor of the positive integer n."""
    if n == 1: return 1
    for i in range(2, int(n**.5) + 1):
        if (n % i) == 0: return i
    return n


# Problem 2
def month_length(month, leap_year=False):
    """Return the number of days in the given month."""
    if month in {"September", "April", "June", "November"}:
        return 30
    elif month in {"January", "March", "May", "July",
                        "August", "October", "December"}:
        return 31
    if month == "February":
        if not leap_year:
            return 28
        else:
            return 29
    else:
        return None


# Problem 3
def operate(a, b, oper):
    """Apply an arithmetic operation to a and b."""
    if type(oper) is not str:
        raise TypeError("oper must be a string")
    elif oper == '+':
        return a + b
    elif oper == '-':
        return a - b
    elif oper == '*':
        return a * b
    elif oper == '/':
        if b == 0:
            raise ZeroDivisionError("division by zero is undefined")
        return a / b
    raise ValueError("oper must be one of '+', '/', '-', or '*'")


# Problem 4
class Fraction(object):
    """Reduced fraction class with integer numerator and denominator."""
    def __init__(self, numerator, denominator):
        if denominator == 0:
            raise ZeroDivisionError("denominator cannot be zero")
        elif type(numerator) is not int or type(denominator) is not int:
            raise TypeError("numerator and denominator must be integers")

        def gcd(a,b):
            while b != 0:
                a, b = b, a % b
            return a
        common_factor = gcd(numerator, denominator)
        self.numer = numerator // common_factor
        self.denom = denominator // common_factor

    def __str__(self):
        if self.denom != 1:
            return "{}/{}".format(self.numer, self.denom)
        else:
            return str(self.numer)

    def __float__(self):
        return self.numer / self.denom

    def __eq__(self, other):
        if type(other) is Fraction:
            return self.numer==other.numer and self.denom==other.denom
        else:
            return float(self) == other

    def __add__(self, other):
        if self.denom == other.denom:
            return Fraction(self.numer + other.numer, self.denom)
        else:
            return Fraction(self.numer * other.denom + self.denom * other.numer,
                                                        self.denom * other.denom)
    def __sub__(self, other):
        if self.denom == other.denom:
            return Fraction(self.numer - other.numer, self.denom)
        else:
            return Fraction(self.numer * other.denom - self.denom * other.numer,
                                                        self.denom*other.denom)
    def __mul__(self, other):
        return Fraction(self.numer*other.numer, self.denom*other.denom)

    def __truediv__(self, other):
        if self.denom*other.numer == 0:
            raise ZeroDivisionError("cannot divide by zero")
        return Fraction(self.numer*other.denom, self.denom*other.numer)

# Problem 6
def count_sets(cards):
    """Return the number of sets in the provided Set hand. 4 possible errors that can be raised in the case the hand isn't actually a hand
    """
    if len(cards) != 12:
        raise ValueError("There are not exactly 12 cards")
    for card in cards:
        if len(card) != 4:
            raise ValueError("One or more cards does not have exactly 4 digits")
    for i in range(0, len(cards)):
        if cards[i] in cards[i+1:len(cards)+1]:
            raise ValueError("There are not 12 unique cards")
        for j in range(0, 4):
            if cards[i][j] not in ['0', '1', '2']:
                raise ValueError("One or more cards has a character other than 0, 1, or 2")
    combination = list(combinations(cards,3))
    num = 0
    for comb in combination:
        a = comb[0]
        b = comb[1]
        c = comb[2]
        if is_set(a,b,c):
            num += 1
    return num

def is_set(a, b, c):
    """Determine if the cards a, b, and c constitute a set.
    """
    isASet = True
    for i in range(0,4):
        if int(a[i])+int(b[i])+int(c[i]) not in [0,3,6]:
            isASet = False
            break

    return isASet
