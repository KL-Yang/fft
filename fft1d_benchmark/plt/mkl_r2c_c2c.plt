# This file compare fftw speed compile with x87, sse and avx by gcc (Debian 4.6.3-14) 4.6.3, on debian wheezy
# icc (ICC) 13.1.3 20130607 none commercial version of icc and mkl
# 
# We can combine two real transform into one complex transform, 
# then use some simple algorithm get the result of two real transform.
#
# It is kind of simple 2 way parellel, while SSE is good for 4 way float parellel, 
# and AVX is good for 8 way float parellel, this demonstrate the potential of speed up.
#

set key right 
set xrange [768 :4096]
set format y "%.1tE%T"
set xlabel "FFT Length"
set ylabel "TPS (Transform Per Second)"
set title "R2C vs C2C Transform by MKL"
plot "../dat/mkl.r2c" using 1:2 with linespoints title "mkl-r2c", \
     "../dat/mkl.c2c" using 1:2 with linespoints title "mkl-c2c", \
     "../dat/mkl.c2c" using 1:($2)*2 with linespoints title "2*mkl-c2c"
pause mouse
