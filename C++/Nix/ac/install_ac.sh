#!/bin/bash

echo
echo "This is used to quickly parse through array-like objects"
echo
echo "This lib has no pre-requisites"
echo "Look for Updates at: github.com/Radicalware"
echo
echo "enjoy!"
echo

x86_64-linux-gnu-g++ -O2 -shared -fPIC ac.cpp ac.h -o libac.so -Wfatal-errors

nm -gC  libac.so > /dev/null 2>&1

chmod 755 libac.so  

sudo cp ./libac.so /usr/local/lib/
sudo cp ./ac.h     /usr/local/include
sudo cp ./ac.cpp   /usr/local/include



