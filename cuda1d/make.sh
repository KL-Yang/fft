#!/bin/bash

#
# $ optirun nvidia-smi
# GeForce GT 630M -> compute compatibility 2.1
#
#
ARCH="arch=compute_20,code=sm_30"

gcc  -std=gnu99 -c utility_cpu.c -W -Wall
nvcc -gencode $ARCH -c utility_gpu.cu

#gcc  -std=gnu99 -c fftw_base.c -W -Wall
#gcc  -std=gnu99 -c fftw_many.c -W -Wall
#nvcc -gencode $ARCH -c drive.cu
