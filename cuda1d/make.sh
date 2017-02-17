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

gcc  $CFLAG -c utility_cpu.c 
gcc  $CFLAG -c fftw_base.c 
gcc  $CFLAG -c fftw_many.c 
gcc  $CFLAG -c valid_base.c
gcc  $CFLAG -c speed_base.c
gcc -o valid_base utility_cpu.o fftw_base.o valid_base.o -lfftw3f -lm
gcc -o speed_base utility_cpu.o fftw_base.o speed_base.o -lfftw3f -lm

nvcc $NFLAG -c utility_gpu.cu

#gcc  -std=gnu99 -c fftw_base.c -W -Wall
#gcc  -std=gnu99 -c fftw_many.c -W -Wall
#nvcc -gencode $ARCH -c drive.cu
