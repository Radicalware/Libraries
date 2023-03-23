#pragma once

#include <stdio.h>
#include <corecrt_math_defines.h>
#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <cstdio>
#include <ciso646>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

using xint = size_t;

#define MAX_UINT 9223372036854775807
#define MAX_INT  2147483647

#define GetWarpID() (threadIdx.x % 32)
#define GetBlockID() (blockIdx.x + (blockIdx.y * blockDim.x) + (blockIdx.z * blockDim.x * blockDim.y))

#define This (*this)

// Inline ___ Functions
#define IHF inline __host__
#define IDF inline __device__
#define IXF inline __host__ __device__
#define IGF static __global__

// Divergent ___ Functions
#define DDF __device__
#define DHF __host__
#define DXF __host__ __device__

// -----------------------------------------------------
__device__ void __syncthreads();
__device__ void __threadfence();
__device__ void __threadfence_block();
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
__device__ int atomicAdd(
    int* address, 
    int val);
__device__ unsigned int atomicAdd(
    unsigned int* address,
    unsigned int val);
__device__ unsigned long long int atomicAdd(
    unsigned long long int* address,
    unsigned long long int val);
__device__ float atomicAdd(
    float* address, 
    float val);
__device__  double atomicAdd(
    double* address, 
    double val);
// -----------------------------------------------------
__device__ int atomicSub(
    int* address, 
    int val);
__device__ unsigned int atomicSub(
    unsigned int* address,
    unsigned int val);
// -----------------------------------------------------
__device__ unsigned int atomicInc(
    unsigned int* address,
    unsigned int val);
// -----------------------------------------------------
__device__ unsigned int atomicDec(
    unsigned int* address,
    unsigned int val);
// -----------------------------------------------------
