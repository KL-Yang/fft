#!/bin/bash

howmany=16
for i in {1..10}; do
    howmany=$(($howmany*2))
    optirun ./speed_cuda 2048 $howmany 100
#    sleep 30    #not burn the GPU too hot!
done
