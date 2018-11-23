#!/bin/bash

echo
echo "This is used to quickly parse through array-like objects"
echo
echo "This lib has no pre-requisites"
echo "Look for Updates at: github.com/Radicalware"
echo
echo "enjoy!"
echo

x86_64-linux-gnu-g++ -O2 -shared -fPIC mc.cpp mc.h -o libmc.so -Wfatal-errors

nm -gC  libmc.so > /dev/null 2>&1

chmod 755 libmc.so  

sudo cp ./libmc.so /usr/local/lib/
sudo cp ./mc.h     /usr/local/include
sudo cp ./mc.cpp   /usr/local/include



