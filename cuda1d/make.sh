#!/bin/bash

#
# $ optirun nvidia-smi
# GeForce GT 630M -> compute compatibility 2.1
#
#
ARCH="arch=compute_20,code=sm_30"

nvcc -gencode $ARCH -c drive.cu
