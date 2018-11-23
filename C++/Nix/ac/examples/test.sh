#!/bin/bash

rm -f ./ex_ac

x86_64-linux-gnu-g++ -O2 ex_ac.cpp -o ex_ac -std=c++17 -Wfatal-errors -lac \
./include/ex_basics.h ./include/fuzz_slice.h ./include/fuzz_dice.h

./ex_ac
