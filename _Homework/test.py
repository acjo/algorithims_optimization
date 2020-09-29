import numpy as np
import random as rand



string = "hello, I would love 30 hotdogs \nand some ketchup"
splitline = string.strip().split('\n')
splitline = [line.strip().split(' ')[::-1] for line in splitline]
for line in splitline:
    newstring = ' '.join(line)
#splitline = [line[::-1] for line in splitline]

print(newstring)
