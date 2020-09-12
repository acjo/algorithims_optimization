#osman_caelan_1_5.py
'''
Caelan Osman
Homework 1.5
September 11 2020
'''
import numpy as np
import random as rand
import time

def matrix_vector_multiplication_for_speed():
    for k in range(1, 12):
        A = np.full((2**k,2**k), rand.randint(0,9))
        B = np.full((2**k, 2**k), rand.randint(0,9))
        x = np.full((2**k,1), rand.randint(0,9))
        start_time_1 = time.time()
        (A@B)@x
        end_time_1 = time.time()
        start_time_2 = time.time()
        A@(B@x)
        end_time_2 = time.time()
        firstTime = end_time_1 - start_time_1
        secondTime = end_time_2 - start_time_2
        timeRatio = firstTime / secondTime
        print("For k = " + str(k) + ",  (AB)x/A(Bx) = " + str(timeRatio))
    print("Clearly shown above the ratio of computation changes quite a bit. From the book we know that the amount of flops required for matrix-matrix and matrix-vector multiplication")
    print("are 2^k * 2^k(2^k-1) and 2^k(2^k-1) flops respectively. Looking at the first computation (AB)x we know that the matrices are both of size 2^k x 2^k. So we know that their")
    print("multiplication is going to take 2^k * 2^k (2^k - 1) flops. This multiplication spits out a new matrix still of size 2^k x 2^k which we multiply a 2^k x 1 matrix by.")
    print("Which costs 2^k(2^k-1) flops. So our total cost of the computation is (2^(3k) + 2^(2k) -2^(k+1)) flops")
    print("\n")
    print("Looking at the second compution A(Bx) we realize it is just two matrix-vector multiplications. Each costing 2^k(2^k-1) For a total of 2(2^(2K) - 2^k) or (2^(2k+1) - 2^(k+1)) flops.")
    print("As we can see for small k either computation will be close to each other. But for large K (AB)x clearly takes much more flops (which translates into more time) than A(Bx).")
    print("The ratio of flops is (2^(3k) + 2^(2k) -2^(k+1)) / (2^(2k+1) - 2^(k+1)) = (4^k+2^k-2)/(2^(k+1)-2)")
    print("Clearly for big k the bottom is way smaller than the top which explains why the time ratio was so big as k increased to 11. Simply said (AB)x takes way more computation than A(Bx).")
    print("\n")

def algebraic_manipulation_for_speed():
	for n in range(1, 12):
    u = np.full((2**n,1), rand.randint(0,9))
    v = np.full((2**n,1), rand.randint(0,9))
    x = np.full((2**n,1), rand.randint(0,9))
    I = np.identity(2**n)
    start_time_1 = time.time()
  	(I + (u @ v.T)) @ x
    end_time_1 = time.time()
    start_time_2 = time.time()
    x + u @ (v.T @ x)
    end_time_2 = time.time()
    firstTime = end_time_1 - start_time_1
    secondTime = end_time_2 - start_time_2
    timeRatio = firstTime / secondTime
    print("For n = " + str(n) + ",  (I+uv^T)x/(x+u(v^Tx)) = " + str(timeRatio))
	print("As n grows large the ratio grows even larger that's because the computation time on top of the ratio for (I+(u@v.T)) @x takes more FLOPs to complete than the computation time on bottom")
    print(" which is x + u @ (v.T @ x). ")

Top:
  u * v^T flops: (2*2^n-1) spits out some number ~2^(n+1)
  I + (some number) flops: 2^2n ~ 2^2n
  (matrix) * x flops: 2**n(2*2**n - 1) spits out another vector ~2^(2n+1)
  total: 2^(n+1)-1 + 2^(2n) + 2^(2n+1) - 2^n  leading order temporal complexity ~ 2^(2n) + 2^(2n+1)
  
  bottom:
  v^T*x flops: (2*2^n-1) spits out some number ~2^(n + 1)
  some number * u flops: 2^n spits out another vector ~ ~2^n
  x + (some number * u) flops: 2^n ~2^n
  total: 2^(n+2) -2  ~leading order temporal complexity ~ 2^(n+2)
