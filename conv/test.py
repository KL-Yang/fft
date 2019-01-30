import numpy
from scipy.linalg import toeplitz
#test.1
a = toeplitz([4,5,6,7], [4,3,2,1])
f = numpy.array([1,2,3,4]).reshape((4,1))
b = numpy.dot(a,f)
print(a)
print(f)
print(b)
#test.2
a = toeplitz([8,9,10,11,12,13,14,15], [8,7,6,5,4,3,2,1])
f = numpy.array([1,2,3,4,5,6,7,8]).reshape((8,1))
b = numpy.dot(a,f)
print(a)
print(f)
print(b)
