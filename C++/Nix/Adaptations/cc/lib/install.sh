#!/bin/bash
[ "$UID" -eq 0 ] || exec sudo bash "$0" "$@"

mkdir -p build
rm -rf ./build/*
cd build

sudo cmake .. -DCMAKE_INSTALL_PREFIX=/opt/Radicalware/Libraries/cpp
sudo make install
echo "Done!"
