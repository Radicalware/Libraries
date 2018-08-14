#!/bin/bash

cp OS.h OS.cpp ./save
x86_64-linux-gnu-g++ -shared -fPIC -O2 OS.h OS.cpp -o libOS.so -std=c++17 -Wfatal-errors -lre


nm -gC libOS.so

chmod 755 libOS.so

sudo cp ./libOS.so /usr/local/lib/
sudo cp ./OS.h /usr/local/include
sudo cp ./OS.cpp /usr/local/include

sudo ldconfig
