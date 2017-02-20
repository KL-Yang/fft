#!/bin/bash

nrlist=`./is2357 16 16384`
for i in $nrlist; do
    howmany=$((16*1024*1024/$i))
    #echo "run through 16*4M data"
    #echo $i, $howmany
    optirun ./speed_cuda $i $howmany 100
done