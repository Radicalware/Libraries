#!/bin/bash

rm -rf ./Project

mkdir Project

mkdir Project/include
mkdir Project/include/re2
mkdir Project/include/util

mkdir Project/src
mkdir Project/src/re2
mkdir Project/src/util

echo "------- Copying Header Files -------"
for i in `ls -l ./re2 | awk '{print $NF}' | grep ".h$"`; do 
    cp -f ./re2/$i ./Project/include/re2
done
for i in `ls -l ./util | awk '{print $NF}' | grep ".h$"`; do 
    cp -f ./util/$i ./Project/include/util
done


echo "------- Copying Source Files -------"
for i in `ls -l ./re2 | awk '{print $NF}' | grep ".cc$"`; do 
    cp -f ./re2/$i ./Project/src/re2/` echo $i | sed 's/cc$/cpp/g'`
done
for i in `ls -l ./util | awk '{print $NF}' | grep ".cc$"`; do 
    cp -f ./util/$i ./Project/src/util/` echo $i | sed 's/cc$/cpp/g'`
done

cp -r $0 ~/Windows/Libraries/Modded_Objects/re2/$0
rm -r ~/Windows/Libraries/Modded_Objects/re2/Project
cp -r ./Project ~/Windows/Libraries/Modded_Objects/re2