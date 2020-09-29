# test_specs.py
"""Python Essentials: Unit Testing.
Caelan Osman
Math 321 sec 3
September 27, 2020
"""

import specs
import pytest


def test_add():
    assert specs.add(1, 3) == 4, "failed on positive integers"
    assert specs.add(-5, -7) == -12, "failed on negative integers"
    assert specs.add(-6, 14) == 8

def test_divide():
    assert specs.divide(4,2) == 2, "integer division"
    assert specs.divide(5,4) == 1.25, "float division"
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.divide(4, 0)
    assert excinfo.value.args[0] == "second input cannot be zero"

# Problem 1: write a unit test for specs.smallest_factor(), then correct it.
def test_smallest_factor():
    '''This function performs a unit test with 100% coverage of the smallest_factor() function in
       specs.py
    '''
    assert specs.smallest_factor(15) == 3, '3 is the smallest prime factor of 15'
    assert specs.smallest_factor(1) == 1, '1 is the smallest prime factor of 1'
    assert specs.smallest_factor(35) == 5, '5 is the smallest prime factor of 35'
    assert specs.smallest_factor(17) == 17, '17 is the smallest prime factor of 17'

# Problem 2: write a unit test for specs.month_length().
def test_month_length():
    '''This function performs a unit test with 100% coverage of the month_length() function in
       specs.py
    '''
    assert specs.month_length("September") == 30, 'There are 30 days in September'
    assert specs.month_length("January") == 31, 'There are 31 days in January'
    assert specs.month_length("February") == 28, 'There are 28 days in a non-leap year February'
    assert specs.month_length("February", True) == 29, 'There are 29 days in a leap year February'
    assert specs.month_length("hello") == None, 'This is not a month'

# Problem 3: write a unit test for specs.operate().
def test_operate():
    '''This function performs a unit test with 100% coverage of the operate() function in
       specs.py
    '''
    assert specs.operate(3,2, '+') == 5, '3+2 is 5 incorrect value returned'
    assert specs.operate(3,2, '-') == 1, '3-2 is 1 incorrect value returned'
    assert specs.operate(3,2, '*') == 6, '3*2 is 6 incorrect value returned'
    assert specs.operate(3,2, '/') == 1.5, '3/2 is 1.5 incorrect value returned'
    with pytest.raises(TypeError) as excinfo:
        specs.operate(3,2,3)
    assert excinfo.value.args[0] == 'oper must be a string'
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.operate(3,0, '/')
    assert excinfo.value.args[0] == 'division by zero is undefined'
    with pytest.raises(ValueError) as excinfo:
        specs.operate(3,4,'[')
    assert excinfo.value.args[0] == "oper must be one of '+', '/', '-', or '*'"


# Problem 4: write unit tests for specs.Fraction, then correct it.
@pytest.fixture
def set_up_fractions():
    frac_1_3 = specs.Fraction(1, 3)
    frac_1_2 = specs.Fraction(1, 2)
    frac_n2_3 = specs.Fraction(-2, 3)
    return frac_1_3, frac_1_2, frac_n2_3

def test_fraction_init(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_3.numer == 1
    assert frac_1_2.denom == 2
    assert frac_n2_3.numer == -2
    frac = specs.Fraction(30, 42)
    assert frac.numer == 5
    assert frac.denom == 7

def test_fraction_str(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert str(frac_1_3) == "1/3"
    assert str(frac_1_2) == "1/2"
    assert str(frac_n2_3) == "-2/3"

def test_fraction_float(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert float(frac_1_3) == 1 / 3.
    assert float(frac_1_2) == .5
    assert float(frac_n2_3) == -2 / 3.

def test_fraction_eq(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 == specs.Fraction(1, 2)
    assert frac_1_3 == specs.Fraction(2, 6)
    assert frac_n2_3 == specs.Fraction(8, -12)


# Problem 5: Write test cases for Set.
