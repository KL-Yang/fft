#!/bin/bash

base=64
for i in {1..128}; do
    howmany=$(($base*$i))
    optirun ./speed_cuda 2048 $howmany 100
    sleep 5
#    sleep 30    #not burn the GPU too hot!
done
