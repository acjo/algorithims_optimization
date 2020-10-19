import numpy as np
import random as rand


def count(string):
    return len(string)

def characters(string):
    char_set = set()
    for character in string:
        char_set.add(character)
    return char_set
def count_characters(string):
    char_count = []
    char_set = characters(string)
    for char in char_set:
        count = string.count(char)
        char_count.append(char + ':')
        char_count.append(count)
    return char_count



#my_str = 'The harder I work, the luckier I get.'
#print(count(my_str))
#print(characters(my_str))
#print(count_characters(my_str))
print(121**562 % 7)
