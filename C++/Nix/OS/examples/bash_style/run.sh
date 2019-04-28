#!/bin/bash


line=$(printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -)

echo $line
mkdir -p build
rm -rf ./build/*
cd build

cmake ..
make

echo $line
../build/ex_bash_style
echo $line