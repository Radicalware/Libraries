#!/bin/bash


mkdir -p build
rm -rf ./build/*
cd build

cmake ..
make

../build/ex_open_n_delete
