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
    return re.compile(r"^(_|[a-zA-Z])[\w_]* *=? *([-]?[\d]*[\.]?[\d]*|[-]?[\d]*[e][-]?[\d]+|['][^']*[']|(_|[a-zA-Z])[\w_]*)$")

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
    #read infile by line
    with open(filename) as infile:
        lines = infile.readlines()

    #regex object to match names, birthdays, emails, phones
    get_name = re.compile(r"^[a-zA-Z]*( )([a-zA-Z]\.( ))?[a-zA-Z]*")
    get_birthday = re.compile(r"[^ ][\d]*\/[\d]*\/[\d]*[^ \n]*")
    get_email = re.compile(r"[^ ]*\@[^ \n]*")
    get_number = re.compile(r"(\d{3}-[^ |\n]*|\(\d*\)[^ \n]*)")

    #regex bject to match the wrong birthdays
    wrong_birthday_1 = re.compile(r"^(\d{1,2}\/)(\d{1,2}\/)(\d{2})$")
    wrong_birthday_2 = re.compile(r"^(\d\/)(\d{1,2}\/)(\d{4})s")
    wrong_birthday_3 = re.compile(r"^(\d{2}\/)(\d\/)(\d{4})$")

    #regex bject to match the wrong numbers
    wrong_number_1 = re.compile(r"^(\d{3})-(\d{3})-(\d{4})$")
    wrong_number_2 = re.compile(r"^(\(\d{3}\))-(\d{3})-(\d{4})$")

    #initialize contact dictionary
    contacts = {}
    for line in lines:

        #intialize and set the name, birthday, and email value
        name = get_name.search(line)
        birthday = get_birthday.search(line)
        email = get_email.search(line)
        number = get_number.search(line)

        #extract string for name birthday and email
        if name is not None:
            name = name.group(0)

        if birthday is not None:
            birthday = birthday.group(0)
            #substitute correct birthday syntax if needed
            birthday = wrong_birthday_1.sub(r"\1\g<2>20\3", birthday)
            birthday = wrong_birthday_2.sub(r"0\g<1>\2\3", birthday)
            birthday = wrong_birthday_3.sub(r"\g<1>0\g<2>\g<3>", birthday)

        if email is not None:
            email = email.group(0)

        if number is not None:
            #substitute correct phone number syntax if needed
            number = number.group(0)
            number = wrong_number_1.sub(r"(\1)\2-\3", number)
            number = wrong_number_2.sub(r"\1\2-\3", number)

        #map the name to the dictionary containing contact info
        contacts[name] = {"birthday" : birthday, "email" : email, "phone" : number}

    return contacts


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





