#!/bin/bash

mkdir -p build
rm -rf ./build/*
cd build

cmake .. 
make

../build/ex_SYS --key-A sub-A-1 sub-A-2 sub-A-3 --key-B sub-B-1 sub-B-2 -a -bcdp 8080  9090 -ef -g 

#  --key-A 
#		sub-A-1 
#		sub-A-2 
#		sub-A-3 
# --key-B 
#		sub-B-1 
#		sub-B-2 
# -a
# -bcdp 8080
# -ef
# -g

