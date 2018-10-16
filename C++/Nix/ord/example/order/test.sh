#!/bin/bash

rm -f ./ex

x86_64-linux-gnu-g++ -O2 ex_ord.cpp -o ex_ord -std=c++17 -Wfatal-errors -lord

./ex_ord
