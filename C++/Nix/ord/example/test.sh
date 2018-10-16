#!/bin/bash

rm -f ./ex

x86_64-linux-gnu-g++ -O2 ex_gr.cpp -o ex -std=c++17 -Wfatal-errors -lord

./ex
