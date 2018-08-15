#!/bin/bash

rm -f ./ver

x86_64-linux-gnu-g++ -O2 ver.cpp -o ver -std=c++17 -Wfatal-errors -L/usr/local/lib -lTimer -I/usr/local/include

./ver	
