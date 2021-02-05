# regular_expressions.py
"""Volume 3: Regular Expressions.
Caelan osman
Math 323 Sec. 2
Feb 3, 2021
"""

import re

# Problem 1
def prob1():
    """Compile and return a regular expression pattern object with the
    pattern string "python".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """

    #return compiled expression
    return re.compile("python")

# Problem 2
def prob2():
    """Compile and return a regular expression pattern object that matches
    the string "^{@}(?)[%]{.}(*)[_]{&}$".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """

    #return compiled expression with meta characters
    return re.compile(r"\^\{@\}\(\?\)\[%\]\{\.\}\(\*\)\[_]\{&\}\$")

# Problem 3
def prob3():
    """Compile and return a regular expression pattern object that matches
    the following strings (and no other strings).

        Book store          Mattress store          Grocery store
        Book supplier       Mattress supplier       Grocery supplier

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """

    return re.compile(r"^(Book|Mattress|Grocery) (store|supplier)$")

# Problem 4
def prob4():
    """Compile and return a regular expression pattern object that matches
    any valid Python identifier.

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile(r"^(_|[a-z]|[A-Z])[\w_]* *=? *([\d]*|[\'][\w]*[\'])$")

# Problem 5
def prob5(code):
    """Use regular expressions to place colons in the appropriate spots of the
    input string, representing Python code. You may assume that every possible
    colon is missing in the input string.

    Parameters:
        code (str): a string of Python code without any colons.

    Returns:
        (str): code, but with the colons inserted in the right places.
    """
    py_code = re.compile(r"^.*(if|elif|else|for|while|try|except|finally|with|def|class)$")

    return py_code.search(code)

# Problem 6
def prob6(filename="fake_contacts.txt"):
    """Use regular expressions to parse the data in the given file and format
    it uniformly, writing birthdays as mm/dd/yyyy and phone numbers as
    (xxx)xxx-xxxx. Construct a dictionary where the key is the name of an
    individual and the value is another dictionary containing their
    information. Each of these inner dictionaries should have the keys
    "birthday", "email", and "phone". In the case of missing data, map the key
    to None.

    Returns:
        (dict): a dictionary mapping names to a dictionary of personal info.
    """

    raise NotImplementedError("Problem 6 Incomplete")


if __name__ == "__main__":

    #problem 2:
    '''
    test_string = "^{@}(?)[%]{.}(*)[_]{&}$"
    expression = prob2()
    print(bool(expression.match(test_string)))
    '''

    #problem 3:
    '''
    test_strings = ["Book store", "Mattress store", "Grocery store", "Book supplier",
                    "Mattress supplier", "Grocery supplier", "Mattress", "supplier store", "M"]
    expression = prob3()
    for string in test_strings:
        print(string + ":", bool(expression.search(string)))
    '''

    #problem 4:
    '''
    pattern = prob4()
    python_identifiers = ["Mouse", "compile", "_123456789", "__x__", "while", "_c__ _",
                          "3rats", "err*r", "sq(x)", "sleep()", " x"]

    for ident in python_identifiers:
        print(ident + ":", bool(pattern.search(ident)))
    '''


    #problem 5:

    py_code = """
              k, i, p = 999, 1, 0
              while k > i
                  i *= 2
                  p += 1
                  if k != 999
                      print("k should not have changed")
                  else
                      pass
             print(p)
"""

    print(prob5(py_code))
