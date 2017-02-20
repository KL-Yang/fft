#!/bin/bash

#
# $ optirun nvidia-smi
# GeForce GT 630M -> compute compatibility 2.1
#
#
ARCH="arch=compute_20,code=sm_21"
CFLAG="-std=gnu99 -W -Wall"
NFLAG="-ccbin=clang-3.8 -gencode $ARCH -Wno-deprecated-gpu-targets"

rm -f *.o

nvcc $NFLAG -c utility.cu 
nvcc $NFLAG -c cuda_fft.cu

gcc  $CFLAG -c fftw_fft.c 
gcc  $CFLAG -c valid_base.c
gcc  $CFLAG -c speed_base.c
nvcc $NFLAG -o valid_base utility.o fftw_fft.o valid_base.o -lcudart -lfftw3f -lm
nvcc $NFLAG -o speed_base utility.o fftw_fft.o speed_base.o -lcudart -lfftw3f -lm


#gcc  -std=gnu99 -c fftw_base.c -W -Wall
#gcc  -std=gnu99 -c fftw_many.c -W -Wall
#nvcc -gencode $ARCH -c drive.cu
