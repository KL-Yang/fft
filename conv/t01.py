import sys
import numpy as np
from scipy.linalg import toeplitz
sys.path.insert(0, './module')
import pytoep as tt

#test.1
a = toeplitz([4,5,6,7], [4,3,2,1])
f = np.array([1,2,3,4], dtype=float).reshape((4,1))
b = np.dot(a,f)
print(a)
print(f)
print(b)

print('test result:')
xa = np.array(range(1,8), dtype=np.float32)
xf = np.array(range(1,5), dtype=np.float32)
print(xa)
print(xf)
xb = tt.mat04(xa, xf)
print(xb)
