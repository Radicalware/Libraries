#!/bin/bash

rm -f ./bash_style

x86_64-linux-gnu-g++ -O2 bash_style.cpp -o bash_style -std=c++17 -Wfatal-errors -lOS

./bash_style
