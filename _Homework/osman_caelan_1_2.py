'''This file includes the algorithms for questions 1.13 and 1.14'''

if __name__ == "__main__":
    def SmallestElementIndex(L):
        min_val = L[0] #2
        min_val_index = 0 #1
        n = len(L) #2
        i = 1 #1
        while i < n: # hit n times executes n-1 times ~n
            if L[i] < min_val: #hit n-1 times 2 primitive operations ~2n (worst case enters every time)
                min_val = L[i] #2 primitive operations n-1 times ~2n
                min_val_index = i #1 operation ~n
            i +=1 #2 operations n-1 ~2n
        return min_val_index

    #leading order temporal complexity is ~8n
    #We store n, i, min_val_index, min_val which are all bounded by log_2(#digits) the List L has
    #is ~n so it the leading order spatial complexity is ~n

    def SelectionSort(L):
        n = len(L) - 1 #3
        i = 0#1
        while i < n: #checked n times failes once executes n-1 times ~n
            index = SmallestElementIndex(L[i:]) + i#call a function with leading order
            #temporal complexity ~8n n-1 times so this is ~8n^2
            temp = L[i]#2
            L[i] = L[index] #2
            L[index] = temp #1
            i += 1 #2 (n-1) times ~2n
        return L
        
#leading order temporal complexity 8n^2
#leading order spatial complexity n^2
