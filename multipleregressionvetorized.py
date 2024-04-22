import numpy as np
import sys
import numpy.random as rand
a=np.zeros(4); print(f"np.zeros(4) : a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a=np.zeros((4,)); print(f"np.zeros(4, ) : a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a=np.random.random_sample(4); print("np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([1,2,3,4])
b=np.array([-1,4,3,2])
print(f"my_dot(a,b) = {my_dot(a,b)}")

a = np.array([1,2,3,4])
b=np.array([-1,4,3,2])
c = np.dot(a,b)
print(f"NumPy 1-D np.dot(a,b) = {c}, np.dot(a,b).shape = {c.shape}")
c = np.dot(b,a)
print(f"NumPy 1-D np.dot(b,a) = {c}, np.dot(a,b).shape = {c.shape}")

import time
np.random.seed(1)
a=np.random.rand(1000000)
b=np.random.rand(1000000)
tic=time.time()
c=np.dot(a,b)
toc=time.time()
print(f"np.dot(a,b) = {c:4f}")
print (f"vectorized version duration:{1000*(toc-tic):.4f} ms")
tic=time.time()
c=my_dot(a,b)
toc =time.time()
print(f"my_dot(a,b) = {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms")
del(a); del(b)

