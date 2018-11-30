#!/bin/bash

rm -f ./file_managment

x86_64-linux-gnu-g++ -O2 file_managment.cpp -o file_managment -std=c++17 -Wfatal-errors -lOS -lre 

./file_managment
echo ---------------------------------------
ls
