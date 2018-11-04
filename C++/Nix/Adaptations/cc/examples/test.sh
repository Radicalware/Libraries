#!/bin/bash

rm -f ./ex_colors

x86_64-linux-gnu-g++ -O2 ex_colors.cpp -o ex_colors -std=c++17 -Wfatal-errors -lcc

./ex_colors