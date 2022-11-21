#pragma once

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#ifndef _uint_
#define _uint_
using uint = size_t;
#endif

#ifndef MAX_UINT
#define MAX_UINT 9223372036854775807
#define MAX_INT  2147483647
#endif  

#ifndef _THIS_
#define This (*this)
#endif

#ifndef _TrueFalse_
#define True  1
#define False 0

#define On  1
#define Off 0
#endif

#ifndef __CUDA_INTELLISENSE__
// -----------------------------------------------------
__device__ void __syncthreads();
// -----------------------------------------------------
__device__ int atomicCAS(
    int* address,
    int compare,
    int val);
__device__ unsigned int atomicCAS(
    unsigned int* address,
    unsigned int compare,
    unsigned int val);
__device__ unsigned long long int atomicCAS(
    unsigned long long int* address,
    unsigned long long int compare,
    unsigned long long int val);
__device__ unsigned short int atomicCAS(
    unsigned short int* address,
    unsigned short int compare,
    unsigned short int val);
// -----------------------------------------------------
__device__ int atomicExch(
    int* address,
    int val);
__device__ unsigned int atomicExch(
    unsigned int* address,
    unsigned int val);
__device__ unsigned long long int atomicExch(
    unsigned long long int* address,
    unsigned long long int val);
__device__ float atomicExch(
    float* address,
    float val);
// -----------------------------------------------------
__device__ int atomicMin(
    int* address, 
    int val);
__device__ unsigned int atomicMin(
    unsigned int* address,
    unsigned int val);
__device__ unsigned long long int atomicMin(
    unsigned long long int* address,
    unsigned long long int val);
__device__ long long int atomicMin(
    long long int* address,
    long long int val);
// -----------------------------------------------------
__device__ int atomicMax(
    int* address, 
    int val);
__device__ unsigned int atomicMax(
    unsigned int* address,
    unsigned int val);
__device__ unsigned long long int atomicMax(
    unsigned long long int* address,
    unsigned long long int val);
__device__ long long int atomicMax(
    long long int* address,
    long long int val);
// -----------------------------------------------------

#endif