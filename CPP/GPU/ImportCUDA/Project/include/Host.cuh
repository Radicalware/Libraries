#pragma once

#include <tuple>

#include "Macros.h"
#include "BasicCUDA.cuh"
#include "Allocate.cuh"
#include "Memory.h"

#include <stdio.h>
#include <cmath>

namespace RA
{
    namespace Host
    {
        TTT static T* CopyHostToDevice(T* FvHostDataPtr, const Allocate& FoAllocate);

        TTT static std::enable_if_t<IsFundamental(T), T*>
                      AllocateArrOnDevice(const xint FnLeng);
        TTT static T* AllocateArrOnDevice(const Allocate& FoAllocate);
        TTT static T* AllocateArrOnDevice(const T* FoHostPtr, const Allocate& FoAllocate);

        template<typename T, typename ...A>
        static T* AllocateObjOnDevice(A&&... Args);
        template<typename B, typename D, typename ...A>
        static B* AllocateObjOnDevice(A&&... Args);

        static void PrintDeviceStats();

        static dim3 GetThreadSquare(const unsigned int FnSquareSize);
        static dim3 GetThreadCube(const unsigned int FnCubeSize);

        static std::tuple<dim3, dim3> GetDimensions3D(const unsigned int FnLeng);
        static std::tuple<dim3, dim3> GetDimensions2D(const unsigned int FnLeng);
        static std::tuple<dim3, dim3> GetDimensions1D(const unsigned int FnLeng);

        static void PopulateStaticNums();

        static unsigned int SnThreadsPerBlock = 0;
        static unsigned int SnThreadsPerWarp = 0;
        static dim3 SvBlock3D;
        static dim3 SvBlock2D;
        static int  SnDeviceCount = 0;

        istatic xint GetThreadsPerBlock() { return RA::Host::SnThreadsPerBlock; }
        istatic dim3 GetBlock3D() { return RA::Host::SvBlock3D; }
        istatic dim3 GetBlock2D() { return RA::Host::SvBlock2D; }
    };
};

TTT T* RA::Host::CopyHostToDevice(T* FvHostDataPtr, const RA::Allocate& FoAllocate)
{
    Begin();
    T* LoDevicePtr = nullptr;
    auto Error = cudaMalloc((void**)&LoDevicePtr, FoAllocate.GetMallocSize());
    if (Error)
        ThrowIt("CUDA Malloc Error: ", cudaGetErrorString(Error));
    Error = cudaMemcpy(LoDevicePtr, FvHostDataPtr, FoAllocate.GetMemCopySize(), cudaMemcpyHostToDevice);
    if (Error)
        ThrowIt("CUDA Memcpy Error: ", cudaGetErrorString(Error));
    return LoDevicePtr;
    Rescue();
}

TTT std::enable_if_t<IsFundamental(T), T*> RA::Host::AllocateArrOnDevice(const xint FnLeng)
{
    Begin();
    auto LvHostDataPtr = RA::SharedPtr<T[]>(FnLeng);
    return RA::Host::CopyHostToDevice<T>(LvHostDataPtr.Ptr(), Allocate(FnLeng, sizeof(T)));
    Rescue();
}

TTT T* RA::Host::AllocateArrOnDevice(const RA::Allocate& FoAllocate)
{
    Begin();
    auto LvHostDataPtr = RA::SharedPtr<T[]>(FoAllocate.GetLength());
    return RA::Host::CopyHostToDevice<T>(LvHostDataPtr.Ptr(), FoAllocate);
    Rescue();
}

TTT T* RA::Host::AllocateArrOnDevice(const T* FoHostPtr, const RA::Allocate& FoAllocate)
{
    Begin();
    T* LoDevicePtr = nullptr;
    auto Error = cudaMalloc((void**)&LoDevicePtr, FoAllocate.GetMallocSize());
    if (Error)
        ThrowIt("CUDA Malloc Error: ", cudaGetErrorString(Error));
    Error = cudaMemcpy(LoDevicePtr, FoHostPtr, FoAllocate.GetMemCopySize(), cudaMemcpyHostToDevice);
    if (Error)
        ThrowIt("CUDA Memcpy Error: ", cudaGetErrorString(Error));
    return LoDevicePtr;
    Rescue();
}

template<typename T, typename ...A>
T* RA::Host::AllocateObjOnDevice(A&& ...Args)
{
    auto LvHostDataPtr = RA::MakeShared<T>(std::forward<A>(Args)...);
    return RA::Host::CopyHostToDevice<T>(LvHostDataPtr.Ptr(), Allocate(1, sizeof(T)));
}

template<typename B, typename D, typename ...A>
B* RA::Host::AllocateObjOnDevice(A&& ...Args)
{
    xp<B> LvHostDataPtr = RA::MakeShared<D>(std::forward<A>(Args)...);
    return RA::Host::CopyHostToDevice<B>(LvHostDataPtr.Ptr(), Allocate(1, sizeof(B)));
}

void RA::Host::PrintDeviceStats()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);


    if (deviceCount == 0)
        printf("No CUDA support device found\n\n");
    else if (deviceCount > 1)
        printf("You have SLi Running!!\n\n");
    else
        printf("You have one video card running\n\n");

    printf("Number of devices:                               %d\n", deviceCount);

    const char* Line = "\n---------------------------------------------------------------------------\n";
    for (int devNo = 0; devNo < deviceCount; devNo++) {

        cudaDeviceProp iProp;
        cudaGetDeviceProperties(&iProp, devNo);
        printf(Line);
        printf("Device %d Model:                                 %s\n", devNo, iProp.name);
        printf("  Number of multiprocessors:                     %d\n", iProp.multiProcessorCount);
        printf("  clock rate:                                    %d\n", iProp.clockRate);
        printf("  Compute capability:                            %d.%d\n", iProp.major, iProp.minor);
        printf("  Total amount of global memory:                 %4.2f KB\n", iProp.totalGlobalMem / 1024.0);
        printf("  Total amount of constant memory:               %4.2f KB\n", iProp.totalConstMem / 1024.0);
        printf("  Total amount of shared memory per block:       %4.2f KB\n", iProp.sharedMemPerBlock / 1024.0);
        printf("  Total amount of shared memory per MP:          %4.2f KB\n", iProp.sharedMemPerMultiprocessor / 1024.0);
        printf("  Total number of registers available per block: %d\n", iProp.regsPerBlock);
        printf("  Warp size:                                     %d\n", iProp.warpSize);
        printf("  Maximum number of threads per block:           %d\n", iProp.maxThreadsPerBlock);
        printf("  Maximum number of threads per multiprocessor:  %d\n", iProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of warps per multiprocessor:    %d\n", iProp.maxThreadsPerMultiProcessor / 32);

        printf("  Maximum Grid size:                            (%d,%d,%d)\n",
            iProp.maxGridSize[0], iProp.maxGridSize[1], iProp.maxGridSize[2]);

        printf("  Maximum block dimension:                      (%d,%d,%d)\n",
            iProp.maxThreadsDim[0], iProp.maxThreadsDim[1], iProp.maxThreadsDim[2]);
    }
    printf(Line);
}

dim3 RA::Host::GetThreadSquare(const unsigned int FnSquareSize)
{
    unsigned int LnX = 1;
    unsigned int LnY = FnSquareSize;
    unsigned int LnNewY = LnY;
    while (LnX < LnY)
    {
        LnNewY /= 2;
        const auto LnNewX = FnSquareSize / LnNewY;
        if (LnNewX * LnNewY != FnSquareSize)
            continue;
        if (LnX * LnY == FnSquareSize
            && (LnY - LnX) < (LnNewY - LnNewX))
            break;
        LnX = LnNewX;
        LnY = LnNewY;
    }
    if (LnX >= LnY)
        return dim3(LnX, LnY);
    return dim3(LnY, LnX);
}

dim3 RA::Host::GetThreadCube(const unsigned int FnCubeSize)
{
    unsigned int LnXY = 1;
    unsigned int LnZ = FnCubeSize;
    unsigned int LnNewZ = LnZ;
    while (LnXY < LnZ)
    {
        LnNewZ /= 2;
        const auto LnXYThreadCount = FnCubeSize / LnNewZ;
        auto LnNewXY = static_cast<unsigned int>(std::sqrt(LnXYThreadCount));
        if (LnNewXY * LnNewXY * LnNewZ != FnCubeSize)
        {
            if (LnNewZ != 1)
                continue;
            return GetThreadSquare(FnCubeSize);
        }
        LnXY = LnNewXY;
        LnZ = LnNewZ;
    }
    return dim3(LnXY, LnXY, LnZ);
}

std::tuple<dim3, dim3> RA::Host::GetDimensions3D(const unsigned int FnLeng)
{
    if (!SnDeviceCount)
        RA::Host::PopulateStaticNums();

    if (SnThreadsPerBlock > (FnLeng / 2))
        return RA::Host::GetDimensions1D(FnLeng);
    if (SnThreadsPerBlock > (FnLeng / 4))
        return RA::Host::GetDimensions2D(FnLeng);
    if (SvBlock3D.x == 0)
        SvBlock3D = GetThreadCube(SnThreadsPerBlock);

    const auto LvBlock = SvBlock3D;
    const auto LnRemainder = (FnLeng % SnThreadsPerBlock) ? 1 : 0;
    auto LnGridThreadCount = FnLeng / SnThreadsPerBlock + LnRemainder;
    if (LnGridThreadCount % 2 != 0)
        LnGridThreadCount++;
    auto LvGrid = (LnGridThreadCount >= 1024)
        ? GetThreadCube(LnGridThreadCount)
        : GetThreadSquare(LnGridThreadCount);
    return std::make_tuple(LvGrid, LvBlock);
}

std::tuple<dim3, dim3> RA::Host::GetDimensions2D(const unsigned int FnLeng)
{
    if (!SnDeviceCount)
        RA::Host::PopulateStaticNums();

    if (SnThreadsPerBlock > (FnLeng / 2))
        return RA::Host::GetDimensions1D(FnLeng);
    if (SvBlock2D.x == 0)
        SvBlock2D = GetThreadSquare(SnThreadsPerBlock);

    const auto LvBlock = SvBlock2D;
    const auto LnRemainder = (FnLeng % SnThreadsPerBlock) ? 1 : 0;
    const auto LvGrid = GetThreadSquare(FnLeng / SnThreadsPerBlock + LnRemainder);
    return std::make_tuple(LvGrid, LvBlock);
}

std::tuple<dim3, dim3> RA::Host::GetDimensions1D(const unsigned int FnLeng)
{
    if (!SnDeviceCount)
        RA::Host::PopulateStaticNums();

    if (FnLeng < SnThreadsPerWarp)
    {
        const dim3 LvBlock = FnLeng;
        const dim3 LvGrid = 1;
        return std::make_tuple(LvGrid, LvBlock);
    }
    const dim3 LvBlock(SnThreadsPerWarp);
    dim3 LvGrid = (FnLeng / LvBlock.x);
    if (LvGrid.x * LvBlock.x < FnLeng)
        LvGrid.x++;
    return std::make_tuple(LvGrid, LvBlock);
}

void RA::Host::PopulateStaticNums()
{
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, 0);
    SnThreadsPerBlock = iProp.maxThreadsPerBlock;
    SnThreadsPerWarp = iProp.warpSize;
    int LnDeviceCount = 0;
    cudaGetDeviceCount(&LnDeviceCount);
    SnDeviceCount = LnDeviceCount;
}

