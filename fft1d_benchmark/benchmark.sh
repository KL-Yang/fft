#! /bin/bash
gcc -o bin/fftben_x87 -std=gnu99 -O3 -ffast-math fftben.c -I/opt/fftw/include -L/opt/fftw/lib -lfftw3f_x87  -lm
gcc -o bin/fftben_sse -std=gnu99 -O3 -msse3 -ffast-math fftben.c -I/opt/fftw/include -L/opt/fftw/lib -lfftw3f_sse2 -lm
gcc -o bin/fftben_avx -std=gnu99 -O3 -mavx -ffast-math fftben.c -I/opt/fftw/include -L/opt/fftw/lib -lfftw3f_avx_fma  -lm
export MKLROOT=/opt/intel/mkl
icc -o bin/fftben_icc -std=c99 -O3 -xAVX  fftben.c -I/opt/fftw/include -L/opt/fftw/lib -lfftw3f_icc  -lm
icc -o bin/fftben_mkl -std=c99 -O3 -xAVX fftben.c -I$MKLROOT/include/fftw -Wl,--start-group -L/opt/intel/lib/intel64/crt $MKLROOT/lib/intel64/libmkl_intel_lp64.a $MKLROOT/lib/intel64/libmkl_sequential.a $MKLROOT/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm
./bin/fftben_x87 100 8192 > ./dat/x87.log
./bin/fftben_sse 100 8192 > ./dat/sse.log
./bin/fftben_avx 100 8192 > ./dat/avx.log
./bin/fftben_icc 100 8192 > ./dat/icc.log
./bin/fftben_mkl 100 8192 > ./dat/mkl.log
./fastsize.py ./dat/x87.log 1 > ./dat/x87.r2c
./fastsize.py ./dat/sse.log 1 > ./dat/sse.r2c
./fastsize.py ./dat/avx.log 1 > ./dat/avx.r2c
./fastsize.py ./dat/icc.log 1 > ./dat/icc.r2c
./fastsize.py ./dat/mkl.log 1 > ./dat/mkl.r2c
./fastsize.py ./dat/x87.log 2 > ./dat/x87.c2c
./fastsize.py ./dat/sse.log 2 > ./dat/sse.c2c
./fastsize.py ./dat/avx.log 2 > ./dat/avx.c2c
./fastsize.py ./dat/icc.log 2 > ./dat/icc.c2c
./fastsize.py ./dat/mkl.log 2 > ./dat/mkl.c2c
