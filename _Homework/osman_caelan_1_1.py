if __name__ == "__main__":

    def subtraction(listA, listB):
        """ Demonstrates the elementary multidigit subtraction algorithm"""
        # Add zeros to the shorter list to align the two lists this way we can loop through
        # the lists without worrying about different lengths.
        delta = abs(len(listA)-len(listB))
        if len(listA) <= len(listB):
            listA = delta * [0] + listA
            #Adding a list of 0s of length delta to list A or list B
        else:
            listB = delta * [0] + listB

        # set initial i value for looping
        i = len(listA) - 1

        #loop through the lists and find the difference between coresponding elements
        while i >= 0:
            if listA[i] - listB[i] < 0: #if the corresponding element listA is smaller than in listB we need to carry so execute
                if i == 0:
                    #in the case that number represented by listA is smaller than the one represented by
                    #ListB we want to make sure we capture the negative.
                    listA[i] -= listB[i]
                else:
                    #This is "carrying the one" operation
                    listA[i-1] -= 1
                    #subtract 1 from the next element
                    listA[i] += 10
                    #add 10 to the current element
                    listA[i] -= listB[i]
                    #subtract the elements
            else:
                #if the element at listA is not smaller than the corresponding element of listB then just subtract the elements
                listA[i] = listA[i] - listB[i]
            #increment our loop variable
            i -= 1
        return listA

#O(n) is both the temporal and spatial complexity. See paper homework for details on explanation. 
