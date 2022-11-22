import numpy as np

a = np.array(
    [ [-4.6, -3.3, -3.1,  1.0,  3.8], 
      [-2.4, -4.5, -1.2, -2.0,  0.9], 
      [-4.4, -1.8,  0.0,  4.0, -0.9], 
      [ 3.2, -0.3, -3.5,  0.0,  0.6], 
      [-4.0,  1.7, -2.1,  2.0,  4.6] ])

b = np.where(a!=0, np.log(abs(a)), 0)
c = np.sign(a)

d = coordinates = np.random.randint(-10, 10, size=(2, 3, 3))
e = np.where(d>0, d, 0)

print(d)

dim = 2
i = np.arange(dim) + 1

f = np.matmul(i,d,axis=0)