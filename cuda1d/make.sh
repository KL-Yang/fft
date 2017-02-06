#!/bin/bash

#
# $ optirun nvidia-smi
# GeForce GT 630M -> compute compatibility 2.1
#
#
ARCH="arch=compute_20,code=sm_30"
CFLAG="-std=gnu99 -W -Wall"
NFLAG="-gencode $ARCH"

gcc  $CFLAG -c utility_cpu.c 
gcc  $CFLAG -c fftw_base.c 
gcc  $CFLAG -c valid.c
gcc -o valid utility_cpu.o fftw_base.o valid.o -lfftw3f -lm

nvcc $NFLAG -c utility_gpu.cu

#gcc  -std=gnu99 -c fftw_base.c -W -Wall
#gcc  -std=gnu99 -c fftw_many.c -W -Wall
#nvcc -gencode $ARCH -c drive.cu
