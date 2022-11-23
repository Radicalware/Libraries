#pragma once

#ifndef __RA_HOST__
#define __RA_HOST__

#include <tuple>

#include "CudaImport.cuh"
#include "RawMapping.h"

#include <stdio.h>
#include <cmath>


namespace RA
{
    class Host
    {
    public:
        template<typename T>
        static __host__ std::enable_if_t<IsFundamental(RemovePtr(T)), void>
            AllocateMemOnDevice(T MoDevicePtr, const uint FnLeng);

        template<typename T>
        static __host__ void AllocateMemOnDevice(T MoDevicePtr, const uint FnLeng, const uint FnUnitByteSize);
        template<typename T>
        static __host__ void AllocateMemOnDevice(T* MoDevicePtr, const uint FnLeng, const uint FnUnitByteSize);
        template<typename T>
        static __host__ void AllocateMemOnDevice(T* MoDevicePtr, const T* MoHostPtr, const uint FnLeng, const uint FnUnitByteSize);

        static __host__ void PrintDeviceStats();

        static __host__ std::tuple<dim3, dim3> GetDimensions3D(const uint FnLeng);
        static __host__ std::tuple<dim3, dim3> GetDimensions2D(const uint FnLeng);
        static __host__ std::tuple<dim3, dim3> GetDimensions1D(const uint FnLeng);

        static uint GetThreadsPerBlock() { return SnThreadsPerBlock; }
        static dim3 GetBlock3D() { return SvBlock3D; }
        static dim3 GetBlock2D() { return SvBlock2D; }
    private:
        static void PopulateStaticNums();

        static uint SnThreadsPerBlock;
        static dim3 SvBlock3D;
        static dim3 SvBlock2D;
        static uint SnThreadsPerWarp;
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
__host__ void RA::Host::AllocateMemOnDevice(T* MoDevicePtr, const uint FnLeng, const uint FnUnitByteSize)
{
    T* LvCPUDataPtr = (T)calloc(FnLeng, FnUnitByteSize);
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
#endif