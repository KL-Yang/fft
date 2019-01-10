#!/bin/bash

flag="-O3 -march=native -W -Wall -std=gnu99"
gcc -c $flag common.c
gcc -o basic $flag main.c common.o -lm

if [ $? -ne 0 ]; then
    exit
fi

./basic -n 4096 -m 12 -r 100000 -a 0 -c -v
