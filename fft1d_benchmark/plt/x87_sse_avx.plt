# This file compare fftw speed compile with x87, sse and avx by gcc (Debian 4.6.3-14) 4.6.3, on debian wheezy

set key right box
set xrange [768 :4096]
set format y "%.1tE%T"
set xlabel "FFT Length"
set ylabel "TPS (Transform Per Second)"
set title "Instruction Set (X87, SSE, AVX) Comparison of R2C Transform"
plot "../dat/x87.r2c" using 1:2 with linespoints title "fftw-x87-gcc", \
     "../dat/sse.r2c" using 1:2 with linespoints title "fftw-sse-gcc", \
     "../dat/avx.r2c" using 1:2 with linespoints title "fftw-avx-gcc"
pause mouse
# This file compare fftw speed compile with x87, sse and avx by gcc (Debian 4.6.3-14) 4.6.3, on debian wheezy

set key right box
set xrange [768 :4096]
set format y "%.1tE%T"
set xlabel "FFT Length"
set ylabel "TPS (Transform Per Second)"
set title "Instruction Set (X87, SSE, AVX) Comparison of C2C Transform"
plot "../dat/x87.c2c" using 1:2 with linespoints title "fftw-x87-gcc", \
     "../dat/sse.c2c" using 1:2 with linespoints title "fftw-sse-gcc", \
     "../dat/avx.c2c" using 1:2 with linespoints title "fftw-avx-gcc"
pause mouse
