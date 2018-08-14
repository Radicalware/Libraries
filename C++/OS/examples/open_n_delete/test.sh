#!/bin/bash

rm -f ./open_n_delete

x86_64-linux-gnu-g++ -O2 open_n_delete.cpp -o open_n_delete -std=c++17 -Wfatal-errors -lOS

./open_n_delete
