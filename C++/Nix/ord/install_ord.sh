#!/bin/bash

echo
echo "This is used to quickly parse through vectors and maps"
echo
echo "re.h  is for string input"
echo "ord.h is for map/vector input"
echo
echo "This lib has no pre-requisites"
echo
echo "enjoy!"
echo

x86_64-linux-gnu-g++ -O2 -shared -fPIC ord.cpp ord.h -o libord.so -Wfatal-errors

nm -gC  libord.so > /dev/null 2>&1

chmod 755 libord.so  

sudo cp ./libord.so /usr/local/lib/
sudo cp ./ord.h     /usr/local/include
sudo cp ./ord.cpp   /usr/local/include



