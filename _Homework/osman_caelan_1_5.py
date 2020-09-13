#osman_caelan_1_5.py
'''
Caelan Osman
Homework 1.5
September 11 2020
'''
import numpy as np
import time

def matrix_vector_multiplication_for_speed():
    for k in range(1, 12):
        A = np.random.randint(0, 10, (2**k, 2**k))
        B = np.random.randint(0, 10, (2**k, 2**k))
        x = np.random.randint(0, 10, (2**k, 2**k))
        start_time_1 = time.time()
        (A@B)@x # number of flops: 2^k*2^k(2*2^k-1) + 2^k(2*2^k-1) = 2^(3k+1) -2^(2k)+ 2^(2k+1) - 2^k
        end_time_1 = time.time()
        start_time_2 = time.time()
        A@(B@x) #number of flops: 2^k(2*2^k-1) + 2^k(2*2^k-1) = 2^(2k+2) -2^(k+1)
        end_time_2 = time.time()
        firstTime = end_time_1 - start_time_1
        secondTime = end_time_2 - start_time_2
        timeRatio = firstTime / secondTime
        print("For k = " + str(k) + ",  (AB)x/A(Bx) = " + str(timeRatio))
    print("\n")
    print("Clearly shown above the ratio of computation changes quite a bit. From the book we know that the amount of FLOPs required for matrix-matrix and matrix-vector multiplication")
    print("are (2^k * 2^k(2^k-1)) and (2^k(2^k-1)) FLOPs respectively. Looking at the first computation (AB)x we know that the matrices are both of size 2^k x 2^k. So we know that their")
    print("multiplication is going to take 2^k * 2^k (2^k - 1) FLOPs. This multiplication spits out a new matrix still of size 2^k x 2^k which we multiply a 2^k x 1 matrix by.")
    print("Which costs 2^k(2^k-1) FLOPs. So our total floating point cost of the computation is (2^(3k+1) + 2^(2k+1)- 2^(2k) -2^k ) flops")
    print("\n")
    print("Looking at the second compution A(Bx) we realize it is just two matrix-vector multiplications. Each costing 2^k(2^k-1) For a total (2^(2k+2) - 2^(k+1)) FLOPs.")
    print("This gives a ratio of for the number of floating point operations which is (2^(3k+1) + 2^(2k+1) - 2(2K) - 2^k) / (2^(2k+2) - 2^(k+1)). And it can be shown that the difference of ratios")
    print("With k+1 and k is 2^(k-1). It's also worth noting that since the denominator of the ratio is always smaller than the top, we know that the ratio will always be greater than one.")
    print("Notice also that the denominator is always small than the numerator which explains why our ratio is always greater than one and increases overall.")
    print("\n")

def algebraic_manipulation_for_speed():
    for n in range(1, 12):
        u = np.random.randint(0, 10, (2**n, 1))
        v = np.random.randint(0, 10, (2**n, 1))
        x = np.random.randint(0, 10, (2**n, 1))
        I = np.identity(2**n)
        start_time_1 = time.time()
        (I + (u @ v.T)) @ x
        #Number of flops: 2^2n  + 2^2n + 2^n(2^(n+1)-1) = 2^(2n+1) + 2^(2n+1) - 2^n = 2^(2n+2) - 2^n ~~~ 2^(2n+2)
        end_time_1 = time.time()
        start_time_2 = time.time()
        x + u @ (v.T @ x) #Numper of flops: 2^(n+1) -1 + 2^n + 2^n = 2^(n+1) - 1 + 2^(n+1) = 2^(n+2) - 1 ~~ 2^(n+2)
        end_time_2 = time.time()
        firstTime = end_time_1 - start_time_1
        secondTime = end_time_2 - start_time_2
        timeRatio = firstTime / secondTime
        print("For n = " + str(n) + ",  (I+uv^T)x/(x+u(v^Tx)) = " + str(timeRatio))
    print("\n")
    print("The FLOPs required for the first computation are calculated as 2^(2n+1) - 2^n and for the second computation we get 2^(n+2) - 1 FLOPs required. Which gives a leading order temporal complexity")
    print("of ~2^(2n+2) for the first method and ~2^(n+2) for the second method. So the ratio has leadin of the first method over the second method has leading order temporal complexity ~2n. Which is why")
    print("The ratio of the two methods increases slowly.")
    print("\n")

