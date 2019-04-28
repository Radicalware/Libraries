#!/bin/bash
[ "$UID" -eq 0 ] || exec sudo bash "$0" "$@"

mkdir -p build
rm -rf ./build/*
cd build

base_dir=$(echo "/opt/Radicalware/Libraries/cpp")
sudo rm -rf $base_dir/code/OS
sudo rm -rf $base_dirinclude/OS
sudo cmake .. -DCMAKE_INSTALL_PREFIX=$base_dir
sudo make install
echo "Done!"
