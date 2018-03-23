import math
import numpy as np

###################################################################
#Testing code 
#Try the iwF(w) deritive thing!
###################################################################

n = 128
f = [14/32.0]   #note: this is aliased at sample interval of 1

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

sig_org = np.zeros(n)
der_org = np.zeros(n)
for i in range(n):
    sig_org[i] = mysig(i, f)
    der_org[i] = myder(i, f)

sig_des = np.zeros(2*n)
for i in range(2*n):
    sig_des[i] = mysig(i/2.0, f)

###################################################################
#Testing code 
#Try the iwF(w) deritive thing!
###################################################################
t2_fft = np.fft.fft(sig_des)
iw_der = np.zeros(2*n, dtype=np.complex_)
for i in range(2*n):
    if i<=n:
        iw_der[i] = 1j*2*math.pi*(i*1.0/n)
    else:
        iw_der[i] = 1j*2*math.pi*((i-2.0*n)/n)
    t2_fft[i] = t2_fft[i]*iw_der[i]
t2_tim = np.fft.ifft(t2_fft)

#decimate and compare with orginal sparse sampling
for i in range(n):
    print "(%9.4f, %9.4f) vs %9.4f" % (t2_tim[2*i].real, t2_tim[2*i].imag, der_org[i])
