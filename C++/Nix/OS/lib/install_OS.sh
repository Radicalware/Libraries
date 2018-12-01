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
echo -e "\e[39menjoy!"
echo

x86_64-linux-gnu-g++ -shared -fPIC -O2 \
	\
	./OS/include/support_os/Dir_Type.h \
	./OS/src/support_os/Dir_Type.cpp \
	\
	./OS/include/support_os/File_Names.h \
	./OS/src/support_os/File_Names.cpp \
	\
	./OS/include/OS.h  \
	./OS/src/OS.cpp  \
	\
	-o libOS.so -std=c++17 -lre  -Wfatal-errors

nm -gC libOS.so  > /dev/null 2>&1

chmod 755 libOS.so

sudo cp ./libOS.so /usr/local/lib/
sudo cp -rf ./OS /usr/local/include


sudo ldconfig
