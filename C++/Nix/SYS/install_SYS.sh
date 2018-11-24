#!/bin/bash
echo
echo "This lib is used similar to python's SYS/SYS libs"
echo
echo "This lib has no pre-requsites!!"
echo 
echo "Source updates can be found at: https://github.com/Radicalware"
echo
echo -e "\e[39menjoy!"
echo

x86_64-linux-gnu-g++ -shared -fPIC -O2 SYS.h SYS.cpp -o libSYS.so -std=c++17 -Wfatal-errors 

nm -gC libSYS.so  > /dev/null 2>&1

chmod 755 libSYS.so

sudo cp ./libSYS.so /usr/local/lib/
sudo cp ./SYS.h /usr/local/include
sudo cp ./SYS.cpp /usr/local/include

sudo ldconfig
