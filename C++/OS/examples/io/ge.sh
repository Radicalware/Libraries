#!/bin/bash


x86_64-linux-gnu-g++ -O2 os_io.cpp -o os_io -std=c++17 -Wfatal-errors -lOS -lord -lre

./os_io -key-A sub-A-1 sub-A-2 sub-A-3 -key-B sub-B-1 sub-B-2 # no sub-B-3

