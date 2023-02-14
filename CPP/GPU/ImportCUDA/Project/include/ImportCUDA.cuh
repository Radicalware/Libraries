#pragma once

// Note: Visual Studio doesn't see preprocessor definitions set by CMake when using CUDA
// The Preprocessor definitions are set and they will compile but your intellisense may be off
// So if you need intellisense, uncomment the next 3 line below (but remember to comment it back out)
//#ifndef UsingNVCC
//#define UsingNVCC
//#endif

#ifndef UsingMSVC

#include "BasicCUDA.cuh"
#include "Host.cuh"
#include "Device.cuh"

#endif