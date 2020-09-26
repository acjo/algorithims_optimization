import numpy as np
import random as rand



string = "hello, I would love 30 hotdogs \n and some ketchup"
splitline = string.strip().split('\n')
splitline = [line.strip().split(' ') for line in splitline]
splitline = [line[::-1] for line in splitline]

splitline = ' '.join('\n'.join(splitline))



print(splitline)
