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

        #2n+1-delta ~2n
        # set initial i value for looping
        i = len(listA) - 1
        #+3
        #log_2(n-1)

        #loop through the lists and find the difference between coresponding elements
        while i >= 0: #~n
            if listA[i] - listB[i] < 0: #if the corresponding element listA is smaller than in listB we need to carry so execute 4n-4 ~4n
                if i == 0: #n-1 ~n
                    #in the case that number represented by listA is smaller than the one represented by
                    #ListB we want to make sure we capture the negative.
                    listA[i] -= listB[i] #4
                else: #n-2
                    #This is "carrying the one" operation
                    listA[i-1] -= 1 #4(n-2) ~4n
                    #subtract 1 from the next element
                    listA[i] += 10 #3(n-2) ~3n
                    #add 10 to the current element
                    listA[i] -= listB[i] #4(n-2) ~ 4n
                    #subtract the elements
            else:
                #if the element at listA is not smaller than the corresponding element of listB then just subtract the elements
                listA[i] -= listB[i]
            #increment our loop variable
            i -= 1 #2(n-2) ~2n
        return listA

#O(n) is both the temporal and spatial complexity. See paper homework for details on explanation.
#2n + n + 4n + n + 4n + 3n + 4n + 2n = 21n
#temporal complexity 21N
#Spatial complexity ~2n
