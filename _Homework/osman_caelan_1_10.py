#osman_caelan_1_10.py


def unimodal(I):
    '''This function returns the larges element in a unimodal list of numbers
       using a binary search method
    '''
    if len(I) != 1: #if length of our list is just one then obviously it's the largest element
        index = len(I)//2 #index
        midpoint = I[index] #midpoint
        if midpoint > I[index - 1]: #if our midpoint is greater than the index minus one recurssively calle unimodal with the second half of the list
            return unimodal(I[index:])
        else:#else recursively call unimodal with the first half of the list
            return unimodal(I[:index])
    else:
        return I[0]

print("We can use the master theorem to prove that this algorithm is in O(log(n)) if n is the size of our list then notice that T(n) satisfies the recursion rule. based on our index we know that ")
print(" b = 2. then our f(n) is just a constant so it is in n^d where d = 0. and a = 1. So using the master theorem we see that T(n) in O(n^0 log(n)) = O(log(n))")
