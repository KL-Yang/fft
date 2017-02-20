#!/bin/bash

#
# $ optirun nvidia-smi
# GeForce GT 630M -> compute compatibility 2.1
#
#
ARCH="arch=compute_20,code=sm_21"
CFLAG="-std=gnu99 -W -Wall -O0"
NFLAG="-ccbin=clang-3.8 -gencode $ARCH -Wno-deprecated-gpu-targets -O0"

rm -f *.o

nvcc $NFLAG -c utility.cu 
nvcc $NFLAG -c cuda_fft.cu
gcc  $CFLAG -c fftw_fft.c 

gcc  $CFLAG -o is2357 is2357.c -lm
nvcc $NFLAG -o valid_base valid_base.c utility.o fftw_fft.o -lcudart -lfftw3f -lm
nvcc $NFLAG -o speed_base speed_base.c utility.o fftw_fft.o -lcudart -lfftw3f -lm
nvcc $NFLAG -o valid_cuda valid_cuda.c utility.o fftw_fft.o cuda_fft.o -lcufft -lcudart -lfftw3f -lm
nvcc $NFLAG -o speed_cuda speed_cuda.c utility.o fftw_fft.o cuda_fft.o -lcufft -lcudart -lfftw3f -lm
