#!/bin/bash

rm -f ./SYS_io

x86_64-linux-gnu-g++ -O2 args.cpp -o args -std=c++17 -Wfatal-errors -lOS -lSYS -lord -lre

./args -key-A sub-A-1 sub-A-2 sub-A-3 -key-B sub-B-1 sub-B-2 # no sub-B-3

