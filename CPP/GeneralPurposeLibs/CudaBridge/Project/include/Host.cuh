#pragma once

#ifndef __RA_HOST__
#define __RA_HOST__

#include <tuple>

#include "Allocate.cuh"
#include "CudaImport.cuh"
#include "Macros.h"

#include <stdio.h>
#include <cmath>


namespace RA
{
    class Host
    {
    public:
        TTT static __host__ std::enable_if_t<IsFundamental(T), T*>
                               AllocateMemOnDevice(const uint FnLeng);
        TTT static __host__ T* AllocateMemOnDevice(const Allocate& FoAllocate);
        TTT static __host__ T* AllocateMemOnDevice(const T* FoHostPtr, const Allocate& FoAllocate);

        static __host__ void PrintDeviceStats();

        static __host__ std::tuple<dim3, dim3> GetDimensions3D(const uint FnLeng);
        static __host__ std::tuple<dim3, dim3> GetDimensions2D(const uint FnLeng);
        static __host__ std::tuple<dim3, dim3> GetDimensions1D(const uint FnLeng);

        template<typename T>
        static __host__ T* CopyHostToDevice(T* FvHostDataPtr, const Allocate& FoAllocate);

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
__host__ T* RA::Host::CopyHostToDevice(T* FvHostDataPtr, const Allocate& FoAllocate)
{
    Begin();
    T* LoDevicePtr = nullptr;
    auto Error = cudaMalloc((void**)&LoDevicePtr, FoAllocate.GetAllocationSize());
    if (Error)
        ThrowIt("CUDA Malloc Error: ", cudaGetErrorString(Error));
    Error = cudaMemcpy(LoDevicePtr, FvHostDataPtr, FoAllocate.GetDataByteSize(), cudaMemcpyHostToDevice);
    if (Error)
    {
        free(FvHostDataPtr);
        ThrowIt("CUDA Memcpy Error: ", cudaGetErrorString(Error));
    }
    return LoDevicePtr;
    Rescue();
}

template<typename T>
__host__ std::enable_if_t<IsFundamental(T), T*> RA::Host::AllocateMemOnDevice(const uint FnLeng)
{
    Begin();
    return RA::Host::AllocateMemOnDevice<T>(Allocate(FnLeng, sizeof(T)));
    Rescue();
}


template<typename T>
__host__ T* RA::Host::AllocateMemOnDevice(const RA::Allocate& FoAllocate)
{
    Begin();
    T* LvHostDataPtr = (T*)calloc(FoAllocate.GetLength() + 1, FoAllocate.GetUnitSize());
    T* LoDevicePtr = RA::Host::CopyHostToDevice<T>(LvHostDataPtr, FoAllocate);
    free(LvHostDataPtr);
    return LoDevicePtr;
    Rescue();
}


template<typename T>
__host__ T* RA::Host::AllocateMemOnDevice(const T* FoHostPtr, const RA::Allocate& FoAllocate)
{
    Begin();
    return RA::Host::CopyHostToDevice<T>(FoHostPtr, FoAllocate);
    Rescue();
}
#endif