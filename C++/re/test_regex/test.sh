#!/bin/bash

rm -f ./ex

x86_64-linux-gnu-g++ -O2 example_regex.cpp -o ex -std=c++17 -Wfatal-errors -lre

./ex
