#!/bin/bash

rm -f ./ex_mc
echo "This example requires ac.h (a Radicalware lib)"
echo
x86_64-linux-gnu-g++ -O2 ex_mc.cpp -o ex_mc -std=c++17 -Wfatal-errors -lac -lmc 

./ex_mc
