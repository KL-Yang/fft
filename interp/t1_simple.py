import sys
import math
import cmath
import numpy as np

#sample interval = 1, interpolate to 256, sample interval = 0.5
#f_nyquist = 1/2, after interpolation, nyquist at 1.0

n=128           #even number of sample, easy for testing
f=[1/32.0, 3/32.0, 18/32.0]
#f=[18/32.0]     #single frequency! alias to 14/32.0

#compute signal [0, 128)
def mysig(x, freq):
    sum_all = 0
    for i in range(len(freq)):
        pha = 2*math.pi*freq[i]
        amp = math.cos(pha*x)
        sum_all = sum_all + amp
    return sum_all

#compute derivative [0, 128)
def myder(x, freq):
    sum_all = 0
    for i in range(len(freq)):
        pha = 2*math.pi*freq[i]
        amp = -1*pha*math.sin(pha*x)
        sum_all = sum_all + amp
    return sum_all

###################################################################
sig_org = np.zeros(n)
der_org = np.zeros(n)
for i in range(n):
    sig_org[i] = mysig(i, f)
    der_org[i] = myder(i, f)
#    print "%6d, %9.4f, %9.4f, %9.4f" % (i, i*1.0, sig_org[i], der_org[i])

fft_org = np.fft.rfft(sig_org)
fnq_org = 1.0/2     #original nyquist frequency
inq_org = n/2       #original nyquist index
for i in range(len(fft_org)):
    freq = fnq_org*i/inq_org
#    print "%6d, %9.4f, %9.4f" % (i, freq, abs(fft_org[i]))

###################################################################
sig_des = np.zeros(2*n)
der_des = np.zeros(2*n)
for i in range(2*n):
    sig_des[i] = mysig(i/2.0, f)
    der_des[i] = myder(i/2.0, f)
#    print "%6d, %9.4f, %9.4f, %9.4f" % (i, i/2.0, sig_des[i], der_des[i])

fft_des = np.fft.rfft(sig_des)
fnq_des = 1.0       #destination nyquist frequency
inq_des = n         #destination nyquist index
for i in range(len(fft_des)):
    freq = fnq_des*(i*1.0/inq_des)
#    print "%6d, %9.4f, %9.4f" % (i, freq, abs(fft_des[i]))

###################################################################
# directly inverse the matrix
###################################################################


###################################################################
# step1: setup matrix A
###################################################################
matA = np.zeros((2*n, 2*n), dtype=np.complex_)
#fill in the W_t^(-1)|_{N\times 2N} part!
for ii in range(n):
    i2 = ii*2   #decimated DFT matrix
    for jj in range(2*n):
        matA[ii][jj] = cmath.exp(1j*2*math.pi*i2*jj/(2*n))
#fill in the W_t^(-1)*D|_{N\times 2N} part!
matRDFT = np.zeros((2*n, 2*n), dtype=np.complex_)
for ii in range(2*n):
    for jj in range(2*n):
        matRDFT[ii][jj] = cmath.exp(1j*2*math.pi*ii*jj/(2*n))
matDIAG = np.zeros((2*n, 2*n), dtype=np.complex_)
for ii in range(n+1): #nyquist frequency is 1.0 in denser sampling
    matDIAG[ii][ii] = 2*math.pi*1j*ii*1.0/n
for ii in range(n+1, 2*n):
    matDIAG[ii][ii] = 2*math.pi*1j*(ii-2.0*n)/n
matTEMP = np.dot(matRDFT, matDIAG)
#the 2nd half is RDFT*DIAG then decimate
for ii in range(n):
    i1 = ii+n
    i2 = ii*2
    for jj in range(2*n):
        matA[i1][jj] = matTEMP[i2][jj]

###################################################################
# step2: setup vector b
###################################################################
vecb = np.zeros(2*n, dtype=np.complex_)
for ii in range(n):
    i1 = ii
    i2 = n+ii
    vecb[i1] = sig_org[ii]
    vecb[i2] = der_org[ii]

###################################################################
# step3: solve the Ax=b, and unwrap time signal
###################################################################
vecx = np.linalg.solve(matA, vecb)
sigx = np.fft.ifft(vecx)*2*n
for i in range(len(sigx)):
    print "%04d %9.4f %9.4f %9.4f" % (i, sigx[i].real, sig_des[i], vecx[i].imag)

#Bingle! problem solved here! Except the DFT matrix normalization issue!
#Once the matrix is inverse, the solution is determinstic!!!

###################################################################
#Testing code 1
#setup DFT matrix and compare with np.fft.fft()
###################################################################
matDFT = np.zeros((2*n, 2*n), dtype=np.complex_)
for ii in range(2*n):
    for jj in range(2*n):
        matDFT[ii][jj] = cmath.exp(-1j*2*math.pi*ii*jj/(2*n))

fft_xxx = np.dot(matDFT, sig_des)
for i in range(len(fft_xxx)):
    freq = fnq_des*(i*1.0/inq_des)
#    print "-%6d, %9.4f, %9.4f" % (i, freq, abs(fft_xxx[i]))

###################################################################
#Testing code 2
#Use the true answer and compute if Ax=y vs b?
###################################################################
fft_ref = np.fft.fft(sig_des)
y = np.dot(matA, fft_ref)/(2*n)
#for i in range(len(y)):
#    print "%4d %9.4f %9.4f %9.4f" % (i, y[i].real, y[i].imag, vecb[i].real)

sys.exit(0)
