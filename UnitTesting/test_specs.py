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
    frac_2_1 = specs.Fraction(2,1)
    return frac_1_3, frac_1_2, frac_n2_3, frac_2_1

def test_fraction_init(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_2_1 = set_up_fractions
    assert frac_1_3.numer == 1
    assert frac_1_2.denom == 2
    assert frac_n2_3.numer == -2
    frac = specs.Fraction(30, 42)
    assert frac.numer == 5
    assert frac.denom == 7
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.Fraction(1,0)
    assert excinfo.value.args[0] == 'denominator cannot be zero'
    with pytest.raises(TypeError) as excinfo:
        specs.Fraction(1.2, 3)
    assert excinfo.value.args[0] == 'numerator and denominator must be integers'

def test_fraction_str(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_2_1 = set_up_fractions
    assert str(frac_1_3) == "1/3"
    assert str(frac_1_2) == "1/2"
    assert str(frac_n2_3) == "-2/3"
    assert str(frac_2_1) == "2"

def test_fraction_float(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_2_1 = set_up_fractions
    assert float(frac_1_3) == 1 / 3.
    assert float(frac_1_2) == .5
    assert float(frac_n2_3) == -2 / 3.

def test_fraction_eq(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_2_1 = set_up_fractions
    assert frac_1_2 == specs.Fraction(1, 2)
    assert frac_1_3 == specs.Fraction(2, 6)
    assert frac_n2_3 == specs.Fraction(8, -12)
    assert specs.Fraction(3,5) == 0.6

def test_fraction_add(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_2_1 = set_up_fractions
    assert frac_1_3 + frac_n2_3 == specs.Fraction(-1,3)
    assert frac_1_2 + frac_2_1 == specs.Fraction(5,2)

def test_fraction_sub(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_2_1 = set_up_fractions
    assert frac_1_3 - frac_n2_3 == specs.Fraction(3,3)
    assert frac_2_1 - frac_1_2 == specs.Fraction(3,2)

def test_fraction_mul(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_2_1 = set_up_fractions
    assert frac_1_3 * frac_1_2 == specs.Fraction(1,6)

def test_fraction_truediv(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_2_1 = set_up_fractions
    assert frac_1_3 / frac_1_2 == specs.Fraction(2,3)
    with pytest.raises(ZeroDivisionError) as excinfo:
        frac_1_2 / specs.Fraction(0,1)
    assert excinfo.value.args[0] == 'cannot divide by zero'

# Problem 5: Write test cases for Set.
@pytest.fixture
def set_up_sets():
    card1 = '1022'
    card2 = '1122'
    card3 = '1222'
    card4 = '1020'
    game = ['0110', '0210', '0010', '1222', '1020', '1200', '2000', '2100', '2010', '2111', '2011', '2001']
    falsegame1 = ['1221', '1222']
    falsegame2 = ['1221', '1222', '1202', '0012', '0101', '0202', '0111', '2222', '0000', '1111', '1212', '0202']
    falsegame3 = ['1221', '12222', '1202', '0012', '0101', '0202', '0111', '2222', '0000', '1111', '1212', '0202']
    falsegame4 = ['1221', '1222', '1202', '0012', '0101', '0202', '0111', '2222', '0000', '1111', '1212', '6120']
    return card1, card2, card3, card4, game, falsegame1, falsegame2, falsegame3, falsegame4

def test_count_sets(set_up_sets):
    card1, card2, card3, card4, game, falsegame1, falsegame2, falsegame3, falsegame4 = set_up_sets
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(falsegame1)
    assert excinfo.value.args[0] == 'There are not exactly 12 cards'
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(falsegame2)
    assert excinfo.value.args[0] == 'There are not 12 unique cards'
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(falsegame3)
    assert excinfo.value.args[0] == 'One or more cards does not have exactly 4 digits'
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(falsegame4)
    assert excinfo.value.args[0] == 'One or more cards has a character other than 0, 1, or 2'
    assert specs.count_sets(game) == 1


def test_is_set(set_up_sets):
    card1, card2, card3, card4, game, falsegame1, falsegame2, falsegame3, falsegame4 = set_up_sets
    assert specs.is_set(card1, card2, card3) == True
    assert specs.is_set(card1, card2, card4) == False
