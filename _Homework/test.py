import numpy as np
import random as rand



u = np.random.randint(0, 10, (3, 1))
v = np.random.randint(0, 10, (3, 1))
print(v @ u.T)

Top:
  u * v^T flops: (2*2^n-1) spits out some number ~2^(n+1)
  I + (some number) flops: 2^2n ~ 2^2n
  (matrix) * x flops: 2**n(2*2**n - 1) spits out another vector ~2^(2n+1)
  total: 2^(n+1)-1 + 2^(2n) + 2^(2n+1) - 2^n  leading order temporal complexity ~ 2^(2n) + 2^(2n+1 )
  
  bottom:
  v^T*x flops: (2*2^n-1) spits out some number ~2^(n + 1)
  some number * u flops: 2^n spits out another vector ~ ~2^n
  x + (some number * u) flops: 2^n ~2^n
  total: 2^(n+2) -2  ~leading order temporal complexity ~ 2^(n+2)
