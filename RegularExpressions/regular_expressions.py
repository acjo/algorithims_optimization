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

    #use groupds to return correct words
    return re.compile(r"^(Book|Mattress|Grocery) (store|supplier)$")

# Problem 4
def prob4():
    """Compile and return a regular expression pattern object that matches
    any valid Python identifier.

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    #set and return regex pattern
    return re.compile(r"^([_a-zA-Z])\w*\s*((=\s*)([_a-zA-Z]*)\w*|(=\s*)([1-9\.]*)|(=\s*)('[_a-zA-Z]*')\w*)?\s*$")

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
    #find the key keywords
    pattern = re.compile(r"(if|elif|else|for|while|try|except|finally|with|def|class).*")
    #use lambda function to add colon
    replacement = lambda x: x.group(0) + ':'
    #return string
    return pattern.sub(replacement, code)

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
    with open(filename, 'r') as infile:
        rawdata = infile.readlines()

    name_exp = re.compile(r"^[a-zA-Z]+(?: [A-Z]\.)? [a-zA-Z]+")
    phone_exp = re.compile(r"(?:1-)?\(?([0-9]{3})\)?-?([0-9]{3})-?([0-9]{4})")
    email_exp = re.compile(r"[\w\.]+@[\w\.]+?\.(?:edu|com|net|org)")
    bday_exp = re.compile(
                    r"(1[0-2]|0?[1-9])/([1-3]\d|0?[1-9])/((?:2[01])?\d{2})")

    def extract(pattern, target):
        results = pattern.findall(target)
        assert len(results) <= 1
        if results:
            return results[0]
        else:
            return None

    data = {}
    for line in rawdata:

        # Check that there is exactly one answer from each expression.
        name = extract(name_exp, line)
        phone = extract(phone_exp, line)
        email = extract(email_exp, line)
        bday = extract(bday_exp, line)

        # Format and structure the data.
        data[name] = {}
        data[name]["email"] = email
        if phone:
            data[name]["phone"] = "({}){}-{}".format(*phone)
        else:
            data[name]["phone"] = None
        if bday:
            month, day, year = bday
            if len(year) == 2:
                year = "20" + year
            if len(month) == 1:
                month = "0" + month
            if len(day) == 1:
                day = "0" + day
            data[name]["birthday"] = "{}/{}/{}".format(month, day, year)
        else:
            data[name]["birthday"] = None
        
    return data


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

    python_paramaters = ["max=-4.", "string= ''", "num_gesses", "300", "is_4=(value==4)", "pattern=r'^one|two fish'", "hello_ = three"]

    print()
    for param in python_paramaters:
        print(param + ":", bool(pattern.search(param)))
    '''


    #problem 5:
    '''
    def helper():
        with open('test.txt') as infile:
            code = infile.read()

        output = prob5(code)
        with open('output.txt', 'w') as outfile:
            outfile.write(output)

    helper()
    '''
    #problem 6:
    '''
    dictionary = prob6()
    print(dictionary)
    '''





