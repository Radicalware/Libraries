#pragma once

#include "CudaImport.cuh"
#include "RawMapping.h"

#include <stdio.h>

using uint = size_t;

namespace RA
{
    namespace Host
    {
        template<typename T>
        __host__ std::enable_if_t<IsFundamental(RemovePtr(T)), void>
            AllocateMemOnDevice(T MoDevicePtr, const uint FnLeng);


        template<typename T>
        __host__ void AllocateMemOnDevice(T MoDevicePtr, const uint FnLeng, const uint FnUnitByteSize);
        template<typename T>
        __host__ void AllocateMemOnDevice(T MoDevicePtr, const uint FnLeng, const uint FnUnitByteSize);
        template<typename T>
        __host__ void AllocateMemOnDevice(T* MoDevicePtr, const T* MoHostPtr, const uint FnLeng, const uint FnUnitByteSize);

        __host__ void PrintDeviceStats();
    };
};


template<typename T>
__host__ std::enable_if_t<IsFundamental(RemovePtr(T)), void> RA::Host::AllocateMemOnDevice(T MoDevicePtr, const uint FnLeng)
{
    constexpr int LnUnitSize = sizeof(RemovePtr(T));

    T LvCPUDataPtr = (T)calloc(FnLeng, LnUnitSize);
    int LnBufferSize = (FnLeng * LnUnitSize) + LnUnitSize;

    cudaMalloc((void**)&MoDevicePtr, LnBufferSize);
    cudaMemcpy(MoDevicePtr, LvCPUDataPtr, FnLeng * LnUnitSize, cudaMemcpyHostToDevice);
}


template<typename T>
__host__ void RA::Host::AllocateMemOnDevice(T MoDevicePtr, const uint FnLeng, const uint FnUnitByteSize)
{
    T LvCPUDataPtr = (T)calloc(FnLeng, FnUnitByteSize);
    int LnBufferSize = (FnLeng * FnUnitByteSize) + FnUnitByteSize;

    cudaMalloc((void**)&MoDevicePtr, LnBufferSize);
    cudaMemcpy(MoDevicePtr, LvCPUDataPtr, FnLeng * FnUnitByteSize, cudaMemcpyHostToDevice);
}

template<typename T>
__host__ void RA::Host::AllocateMemOnDevice(T* MoDevicePtr, const T* MoHostPtr, const uint FnLeng, const uint FnUnitByteSize)
{
    int LnBufferSize = (FnLeng * FnUnitByteSize)+ sizeof(uint);

    cudaMalloc((void**)&MoDevicePtr, LnBufferSize);
    auto LeFinishState = cudaMemcpy(MoDevicePtr, MoHostPtr, FnLeng * FnUnitByteSize, cudaMemcpyHostToDevice);
    printf("Finish State: %d\n", LeFinishState);
}