#! /bin/bash
# gcc 4.6/4.7 seems not support some of the avx intrinsic, so I used icc here
icc -o validsse -std=gnu99 -msse3 simple_sse.c   -I/opt/fftw/include -L/opt/fftw/lib -lfftw3f -lm
icc -o validavx -std=gnu99 -mavx  simple_avx.c   -I/opt/fftw/include -L/opt/fftw/lib -lfftw3f -lm
icc -o speed srme3d.c -std=gnu99 -msse3 -O3      -I/opt/fftw/include -L/opt/fftw/lib -lfftw3f_avx_fma -lm
