#!/bin/bash
echo
echo "This lib is used similar to python's SYS/OS libs"
echo
echo "This lib requires re.h"
echo "re.h has no pre-requisites"
echo "I use the ord.h lib in one of the OS.h examples"
echo 
echo "re.h, ord.h & OS.h can be found at: https://github.com/Radicalware"
echo
echo -e "\e[33mALERT!!! THIS ONLY HAS LINUX SUPPORT AS OF RIGHT NOW"
echo -e "All of my future libs will start on Windows"
echo
echo -e "\e[39menjoy!"
echo

x86_64-linux-gnu-g++ -shared -fPIC -O2 OS.h OS.cpp -o libOS.so -std=c++17 -Wfatal-errors -lre

nm -gC libOS.so  > /dev/null 2>&1

chmod 755 libOS.so

sudo cp ./libOS.so /usr/local/lib/
sudo cp ./OS.h /usr/local/include
sudo cp ./OS.cpp /usr/local/include

sudo ldconfig
