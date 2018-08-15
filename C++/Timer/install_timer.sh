#!/bin/bash

echo
echo "Timer for benchmarking built by learn-cpp.com"
echo
echo "enjoy!"
echo

x86_64-linux-gnu-g++ -shared -O2 -fPIC Timer.cpp Timer.h -o libTimer.so -Wfatal-errors

nm -gC libTimer.so > /dev/null 2>&1

chmod 755 libTimer.so  

sudo cp ./libTimer.so /usr/local/lib/
sudo cp ./Timer.h     /usr/local/include
sudo cp ./Timer.cpp   /usr/local/include

sudo ldconfig
