#!/bin/bash

echo
echo "This lib is used to easily install an interator for your array based objects"
echo
echo "This lib has no pre-requisites"
echo
echo "enjoy!"
echo

x86_64-linux-gnu-g++ -shared -O2 -fPIC Iterator.cpp Iterator.h -o libIterator.so -Wfatal-errors

nm -gC libIterator.so > /dev/null 2>&1

chmod 755 libIterator.so  

sudo cp ./libIterator.so /usr/local/lib/
sudo cp ./Iterator.h /usr/local/include
sudo cp ./Iterator.cpp /usr/local/include

sudo ldconfig
 
