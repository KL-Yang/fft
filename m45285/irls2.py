import sys
import numpy as np

##################################################
# Ax=b L_1 solver
# 
# 0 W=I
# 1 solve y for AWy=b
# 2 x = Wy, W=diag(x)
#
##################################################
def l1_solve(A, b, niter):
    n = A.shape[1]                  #m by n
    W = np.diag(np.ones(n))   #I matrix
    for it in range(niter):
        B = np.dot(A, W)
        y = np.linalg.lstsq(B, b)[0]
        x = np.dot(W, y)
        e = np.linalg.norm(x, 1)
        print("Iter %4d: norm=%f" % (it, e))
        print(x)
        W = np.diagflat(x)
    return x


##################################################
# Ax=b
##################################################
A = np.array([1,0,0, 0, 1, 0.5]).reshape((2,3))
b = np.array([1, 1]).reshape((2, 1))
W = np.diag([1,1,4])
print("A=:")
print(A)
print("b=:")
print(b)
print("W=:")
print(W)

##################################################
# Ax=b L_2 least square solution 
##################################################
x = np.linalg.lstsq(A, b)[0]
print("Ax=b L_2 solution:")
print(x)

#x = l1_solve(A, b, 20)
#print("Ax=b L_1 solution:")
#print(x)

##################################################
# AWx=b L_2 least square solution 
##################################################
x = np.linalg.lstsq(np.dot(A,W), b)[0]
print("AWx=b L_2 solution:")
print(x)

#AW = np.dot(A,W)
#x = l1_solve(AW, b, 20)
#print("AWx=b L_1 solution:")
#print(x)

######################################################################################
def irls_m45285(A, b, p=1.1, K=0.8, KK=20):
    pk = 2
    x = np.linalg.lstsq(A, b)[0]
    E = []
    for k in range(KK):
        if p>=2:
            pk = min([p, K*pk])
        else:
            pk = max([p, K*pk])
        W  = np.diagflat(abs(x)**((2-pk)/2.0+1E-4))
        AW = np.dot(A, W)
        x2 = np.linalg.solve(np.dot(AW, AW.transpose()), b)
        x1 = np.dot(np.dot(W, AW.transpose()), x2)
        if p>=2:
            q = 1/(pk-1.0)
            x = q*x1 + (1-q)*x
            nn = p
        else:
            x = x1
            nn=1
        E.append(np.linalg.norm(x, nn))
        print("m45285[%4d]: norm=%f" %((k, E[-1])))
    return x

x = irls_m45285(A, b, 0.1)
print("X=", x)
sys.exit(0)

##################################################
# Note the above scheme seems work fine
# another solutions mentioned by title
# 1. Least Squares with Examples in Signal Processing
# by Ivan Selesnick of NYU-Poly
# The result seems to be wrong!
# 2. One different version is by OpenStax-CNX m45285
# Iterative Reweighted Least Squares
#
##################################################
def nyu_wl2_solve(A, W, b):
    #first solve AW^{-1}A^Tx=b
    Wi = np.linalg.inv(W) #inverse of diagonal is trival
    A0 = np.dot(A, Wi)
    A0 = np.dot(A0, A.transpose())
    #x0 = np.linalg.lstsq(A0, b)[0]
    x0 = np.linalg.solve(A0, b)
    #x = W^{-1}A^Tx0
    T0 = np.dot(Wi, A.transpose())
    x = np.dot(T0, x0)
    return x

def nyu_wl1_solve(A, b, niter):
    n = A.shape[1]
    W = np.diag(np.ones(n))
    for it in range(niter):
        x = nyu_wl2_solve(A, W, b)
        print("Iter %4d:" %it)
        print(x)
        W = np.diagflat((abs(x)**0.5))
    return x

Wt = np.diag([1,1,4])
x = nyu_wl2_solve(A, Wt, b)
print("Ax=b Weighted L_2 NYU solution:")
print(x)


x = nyu_wl1_solve(A, b, 20)
print("Ax=b Weighted L_1 NYU solution:")
print(x)


sum=0
for i in range(len(x)):
    print("wt=%f, x=%f" % (Wt[i][i], x[i]))
    sum = sum+Wt[i][i]*x[i]*x[i]
test = x[0]*x[0]+x[1]*x[1]+4*x[2]*x[2]
print("sum=", sum)
print(test)


