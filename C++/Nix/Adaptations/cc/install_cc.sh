#!/bin/bash

echo
echo "This lib is programed by Ihor Kalnytskyi"
echo "It was modded by Scourge"
echo "original was termcolor.h modded version is cc.h"
echo
echo "This lib has no pre-requisites"
echo
echo "enjoy!"
echo

x86_64-linux-gnu-g++ -shared -O2 -fPIC cc.cpp cc.h -o libcc.so -Wfatal-errors

nm -gC libcc.so > /dev/null 2>&1

chmod 755 libcc.so  

sudo cp ./libcc.so /usr/local/lib/
sudo cp ./cc.h /usr/local/include
sudo cp ./cc.cpp /usr/local/include

sudo ldconfig
