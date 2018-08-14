#!/bin/bash


# ge.sh ./cpp_file

file=$(echo $1 | sed 's/\.cpp$//g')

x86_64-linux-gnu-g++ -O2 $file.cpp -o $file -std=c++17 -Wfatal-errors -lOS -lre 

./$file
