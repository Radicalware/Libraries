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
        template<typename T>
        static __host__ std::enable_if_t<IsFundamental(T), void>
            AllocateMemOnDevice(T* FoDevicePtr, const uint FnLeng);
        template<typename T>
        static __host__ void AllocateMemOnDevice(T* FoDevicePtr, const Allocate& FoAllocate);
        template<typename T>
        static __host__ void AllocateMemOnDevice(T* FoDevicePtr, const T* FoHostPtr, const Allocate& FoAllocate);

        static __host__ void PrintDeviceStats();

        static __host__ std::tuple<dim3, dim3> GetDimensions3D(const uint FnLeng);
        static __host__ std::tuple<dim3, dim3> GetDimensions2D(const uint FnLeng);
        static __host__ std::tuple<dim3, dim3> GetDimensions1D(const uint FnLeng);

        template<typename T>
        static __host__ void CopyHostToDevice(void** FoDevicePtr, const T* FvHostDataPtr, const Allocate& FoAllocate);

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
__host__ void RA::Host::CopyHostToDevice(void** FoDevicePtr, const T* FvHostDataPtr, const Allocate& FoAllocate)
{
    Begin();
    auto Error = cudaMalloc((void**)&FoDevicePtr, FoAllocate.GetAllocationSize());
    if (Error)
        ThrowIt("CUDA Malloc Error: ", cudaGetErrorString(Error));
    Error = cudaMemcpy(FoDevicePtr, FvHostDataPtr, FoAllocate.GetDataByteSize(), cudaMemcpyHostToDevice);
    if (Error)
    {
        free(FvHostDataPtr);
        ThrowIt("CUDA Memcpy Error: ", cudaGetErrorString(Error));
    }
    Rescue();
}

template<typename T>
__host__ std::enable_if_t<IsFundamental(T), void> RA::Host::AllocateMemOnDevice(T* FoDevicePtr, const uint FnLeng)
{
    Begin();
    constexpr int LnUnitSize = sizeof(T);
    Allocate LvAllocate(FnLeng, LnUnitSize);
    T* LvHostDataPtr = (T)calloc(FnLeng, LvAllocate.GetUnitSize());
    RA::Host::CopyHostToDevice(FoDevicePtr, LvHostDataPtr, LvAllocate);
    free(LvHostDataPtr);
    Rescue();
}


template<typename T>
__host__ void RA::Host::AllocateMemOnDevice(T* FoDevicePtr, const RA::Allocate& FoAllocate)
{
    Begin();
    T* LvHostDataPtr = (T)calloc(FnLeng, FoAllocate.GetUnitSize());
    RA::Host::CopyHostToDevice(FoDevicePtr, LvHostDataPtr, FoAllocate);
    free(LvHostDataPtr);
    Rescue();
}

template<typename T>
__host__ void RA::Host::AllocateMemOnDevice(T* FoDevicePtr, const T* FoHostPtr, const RA::Allocate& FoAllocate)
{
    Begin();
    RA::Host::CopyHostToDevice(FoDevicePtr, FoHostPtr, FoAllocate);
    Rescue();
}
#endif