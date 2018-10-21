#!/bin/bash

rm -f ./slice_n_dice

x86_64-linux-gnu-g++ -O2 slice_n_dice.cpp -o slice_n_dice -std=c++17 -Wfatal-errors -lord fuzz_slice.h fuzz_dice.h

./slice_n_dice

