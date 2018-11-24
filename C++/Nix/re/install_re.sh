#!/bin/bash

echo
echo "This lib is used similar to python's re lib for regex"
echo
echo "re.h  is for string input manipulation"
echo "This lib has no pre-requisites"
echo
echo "enjoy!"
echo

x86_64-linux-gnu-g++ -shared -O2 -fPIC re.cpp re.h -o libre.so -Wfatal-errors

nm -gC libre.so > /dev/null 2>&1

chmod 755 libre.so  

sudo cp ./libre.so /usr/local/lib/
sudo cp ./re.h /usr/local/include
sudo cp ./re.cpp /usr/local/include

sudo ldconfig
