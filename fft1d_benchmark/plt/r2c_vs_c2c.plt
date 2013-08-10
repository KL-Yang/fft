# This file compare fftw speed compile with x87, sse and avx by gcc (Debian 4.6.3-14) 4.6.3, on debian wheezy
# icc (ICC) 13.1.3 20130607 none commercial version of icc and mkl
# 
# We can combine two real transform into one complex transform, 
# then use some simple algorithm get the result of two real transform.
#
# It is kind of simple 2 way parellel, while SSE is good for 4 way float parellel, 
# and AVX is good for 8 way float parellel, this demonstrate the potential of speed up.
#

#set terminal latex
set term tikz color solid
set output "latex/avx_r2c_c2c.tex"
set key right 
set xrange [768 :4096]
set format y "%.1tE%T"
set xlabel "FFT Length"
set ylabel "TPS (Transform Per Second)"
set title "R2C vs C2C Transform by FFTW (AVX)"
plot "../dat/avx.r2c" using 1:2 with linespoints title "avx-r2c", \
     "../dat/avx.c2c" using 1:2 with linespoints title "avx-c2c", \
     "../dat/avx.c2c" using 1:($2)*2 with linespoints title "2*avx-c2c"
#pause mouse

set output "latex/sse_r2c_c2c.tex"
set title "R2C vs C2C Transform by FFTW (SSE)"
plot "../dat/sse.r2c" using 1:2 with linespoints title "sse-r2c", \
     "../dat/sse.c2c" using 1:2 with linespoints title "sse-c2c", \
     "../dat/sse.c2c" using 1:($2)*2 with linespoints title "2*sse-c2c"
#pause mouse

set output "latex/x87_r2c_c2c.tex"
set title "R2C vs C2C Transform by FFTW (X87)"
plot "../dat/x87.r2c" using 1:2 with linespoints title "x87-r2c", \
     "../dat/x87.c2c" using 1:2 with linespoints title "x87-c2c", \
     "../dat/x87.c2c" using 1:($2)*2 with linespoints title "2*x87-c2c"
#pause mouse
