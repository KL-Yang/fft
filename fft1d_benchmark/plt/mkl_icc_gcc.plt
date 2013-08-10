# This file compare fftw speed compile with x87, sse and avx by gcc (Debian 4.6.3-14) 4.6.3, on debian wheezy

set key right box
set xrange [768 :4096]
set format y "%.1tE%T"
set xlabel "FFT Length"
set ylabel "TPS (Transform Per Second)"
set title "FFTW-GCC, FFTW-ICC and FFTW-MKL R2C Transform"
plot "../dat/mkl.r2c" using 1:2 with linespoints title "icc-mkl", \
     "../dat/icc.r2c" using 1:2 with linespoints title "avx-icc-fftw", \
     "../dat/avx.r2c" using 1:2 with linespoints title "avx-gcc-fftw"
pause mouse
# This file compare fftw speed compile with x87, sse and avx by gcc (Debian 4.6.3-14) 4.6.3, on debian wheezy

set title "FFTW-GCC, FFTW-ICC and FFTW-MKL C2C Transform"
plot "../dat/mkl.c2c" using 1:2 with linespoints title "icc-mkl", \
     "../dat/icc.c2c" using 1:2 with linespoints title "avx-icc-fftw", \
     "../dat/avx.c2c" using 1:2 with linespoints title "avx-gcc-fftw"
pause mouse
