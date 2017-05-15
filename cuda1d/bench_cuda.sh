#!/bin/bash

nrlist=`./is2357 16 32768`
for i in $nrlist; do
    howmany=$((32*1024*1024/$i))
    howtail=$(($howmany % 64))
    howmany=$(($howmany - $howtail))
    optirun ./speed_cuda $i $howmany 100
    sleep 1m    #not burn the GPU too hot!
done
