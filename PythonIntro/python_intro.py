# python_intro.py
"""Python Essentials: Introduction to Python.
Calean Osman<Name>
Math 321 Section 3
September 3, 2020
"""
'''Problem 1:'''
print("Hello, World!")

'''problem 2:'''
def sphere_volume(r):
    """This returns the volume of a sphere"""
    PI = 3.14159
    vol = 4/3 * PI * r ** 3
    return(vol)

'''Problem 3:'''
def isolate(a, b, c, d, e):
    """This function seperates the first 3 paramters by 5 spaces and
    the last two by a single space"""
    print(a, b, c, sep = '     ', end = ' ')
    print(d, e, sep = ' ')
    return

'''Problem 4:'''
def first_half(strng):
    """This function prints the first half of a word (not including
        the middle character if the word has an odd length"""
    length = len(strng)
    subset = length // 2
    new_strng = strng[:subset]
    return new_strng

def backward(strng):
    """This function uses slicing to print a word backward. The step
        is negative 'reading' the word backward."""
    return strng[::-1]

'''Problem 5:'''
def list_ops():
    name_list = ["bear", "ant", "cat", "dog"]
    """This function accepts a list and then performs the following functions on that list"""
    name_list.append("eagle") #add's eagle to the end of the list
    name_list[2] = "fox" #make element at index 2 "fox" instead of "catfp"
    name_list.pop(1) #returns and removes item at 1
    name_list.sort(reverse=True) #sorts the list in revers alphabetical order
    #n becomes the index where eagle is found and then we use n to replace the nth element with hawk
    n = name_list.index("eagle")
    name_list[n] = "hawk"
    #n becomes the index of the final element then we add hunter to the final element
    n = len(name_list) - 1
    name_list[n] = name_list[n] + "hunter"
    return name_list

'''Problem 6:'''
def pig_latin(word):
    """This function accepts a word and translates it to pig latin"""
    vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
    if word[0] in vowels:
        word += "hay"
    elif word[0] not in vowels:
        word += word[0]
        word = word[1:len(word)] + "ay"
    return word

'''Problem 7:'''
def palindrome():
    """This function returns the largest palindrome found by multiplying
        two three digit numbers"""
    palinList = []
    for x in range(100,1000):
        for y in range(100,1000):
            num = x * y
            if str(num) == str(num)[::-1]:
                if num not in palinList:
                    palinList.append(num)
    palinList.sort(reverse = True)
    return palinList[0]

'''Problem 8:'''
def alt_harmonic(i):
    """This function computes the alternating harmonic series for 'n' terms
        using list comprehension"""
    approx = [((-1) ** (n+1)) / n for n in range(1,i+1)]
    altharm = sum(approx)
    approx.clear()
    return altharm
