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
        struct Config
        {
            istatic int  SnThreadsPerBlock = 0;
            istatic int  SnThreadsPerWarp = 0;
            istatic dim3 SvBlock3D = dim3(0, 0, 0);
            istatic dim3 SvBlock2D = dim3(0, 0, 0);
            istatic int  SnDeviceCount = 0;

            istatic const int SnReq3D = 4;
            istatic const int SnReq2D = 2;
        };

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

        istatic xint GetThreadsPerBlock() { return Config::SnThreadsPerBlock; }
        istatic dim3 GetBlock3D() { return Config::SvBlock3D; }
        istatic dim3 GetBlock2D() { return Config::SvBlock2D; }

        static void PrintGridBlockDims(const dim3 FvGrid, const dim3 FvBlock);
        static xstring GetGridBlockStr(const dim3 FvGrid, const dim3 FvBlock);
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
    if (!Config::SnDeviceCount)
        RA::Host::PopulateStaticNums();

    constexpr auto Mult = [](const dim3& FoDim) ->xint {
        return static_cast<xint>(FoDim.x * FoDim.y * FoDim.z);
    };
    constexpr auto DecreaseVals = [](dim3& FoDim) -> void
    {
        if (FoDim.z >= FoDim.x && FoDim.z >= FoDim.y)
            --FoDim.z;
        else if (FoDim.y >= FoDim.x)
            --FoDim.y;
        else
            --FoDim.x;
    };
    constexpr auto IncreaseVals = [](dim3& FoDim) -> void
    {
        if (FoDim.z < FoDim.x && FoDim.z < FoDim.y)
            ++FoDim.z;
        else if (FoDim.y < FoDim.x)
            ++FoDim.y;
        else
            ++FoDim.x;
    };

    auto Ln6Split = RA::Pow(FnLeng, 1.0 / 6.0);
    while (RA::Pow(Ln6Split, 6.0) < FnLeng)
        Ln6Split++;

    const auto LnTargetSplit = RA::Pow(Ln6Split, 3);

    auto LvBlock = dim3(32, 1, 1);
    auto LvGrid = dim3(Ln6Split, Ln6Split, Ln6Split);

    auto LnLastY = 1;
    auto LnLastZ = 1;
    while (Mult(LvBlock) < LnTargetSplit)
    {
        if (LvBlock.z < LvBlock.x && LvBlock.z < LvBlock.y)
            ++LvBlock.z;
        else if (LvBlock.y < LvBlock.x)
            ++LvBlock.y;
        else
        {
            LvBlock.x += 32;
            LvBlock.y = LnLastY;
            LvBlock.z = LnLastZ;

            if (LvBlock.z < LvBlock.y)
                LvBlock.z++; // mult by 32 as X
            else
                LvBlock.y++; // mult by 32 as X
            LnLastY = LvBlock.y;
            LnLastZ = LvBlock.z;
        };
    }


    while ((Mult(LvGrid) * Mult(LvBlock)) > FnLeng)
    {
        DecreaseVals(LvGrid);
    }
    while ((Mult(LvGrid) * Mult(LvBlock)) < FnLeng)
    {
        IncreaseVals(LvGrid);
    }
    return std::make_tuple(LvGrid, LvBlock);
}

std::tuple<dim3, dim3> RA::Host::GetDimensions2D(const unsigned int FnLeng)
{
    if (!Config::SnDeviceCount)
        RA::Host::PopulateStaticNums();

    constexpr auto Mult = [](const dim3& FoDim) ->xint {
        return static_cast<xint>(FoDim.x * FoDim.y * FoDim.z);
    };
    constexpr auto DecreaseVals = [](dim3& FoDim) -> void
    {
        if (FoDim.y >= FoDim.x)
            --FoDim.y;
        else
            --FoDim.x;
    };
    constexpr auto IncreaseVals = [](dim3& FoDim) -> void
    {
        if (FoDim.y < FoDim.x)
            ++FoDim.y;
        else
            ++FoDim.x;
    };

    auto Ln4Split = RA::Pow(FnLeng, 1.0 / 4.0);
    while (RA::Pow(Ln4Split, 4.0) < FnLeng)
        Ln4Split++;

    const auto LnTargetSplit = RA::Pow(Ln4Split, 2);

    auto LvBlock = dim3(32, 1, 1);
    auto LvGrid = dim3(Ln4Split, Ln4Split, 1);

    auto LnLastY = 1;
    while (Mult(LvBlock) < LnTargetSplit)
    {
        if (LvBlock.y < LvBlock.x)
            ++LvBlock.y;
        else
        {
            LvBlock.x += 32;
            LvBlock.y = LnLastY;

            LvBlock.y++; // mult by 32 as X
            LnLastY = LvBlock.y;
        };
    }


    while ((Mult(LvGrid) * Mult(LvBlock)) > FnLeng)
    {
        DecreaseVals(LvGrid);
    }
    while ((Mult(LvGrid) * Mult(LvBlock)) < FnLeng)
    {
        IncreaseVals(LvGrid);
    }
    return std::make_tuple(LvGrid, LvBlock);
}

std::tuple<dim3, dim3> RA::Host::GetDimensions1D(const unsigned int FnLeng)
{
    if (!Config::SnDeviceCount)
        RA::Host::PopulateStaticNums();

    if (FnLeng < Config::SnThreadsPerWarp)
    {
        const dim3 LvBlock = FnLeng;
        const dim3 LvGrid = 1;
        return std::make_tuple(LvGrid, LvBlock);
    }
    const dim3 LvBlock(Config::SnThreadsPerWarp);
    dim3 LvGrid = (FnLeng / LvBlock.x);
    if (LvGrid.x * LvBlock.x < FnLeng)
        LvGrid.x++;
    return std::make_tuple(LvGrid, LvBlock);
}

void RA::Host::PopulateStaticNums()
{
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, 0);
    Config::SnThreadsPerBlock = iProp.maxThreadsPerBlock;
    Config::SnThreadsPerWarp = iProp.warpSize;
    int LnDeviceCount = 0;
    cudaGetDeviceCount(&LnDeviceCount);
    Config::SnDeviceCount = LnDeviceCount;
}

void RA::Host::PrintGridBlockDims(const dim3 FvGrid, const dim3 FvBlock)
{
    RA::Print("Grid/Block: ", GetGridBlockStr(FvGrid, FvBlock));
}


xstring RA::Host::GetGridBlockStr(const dim3 FvGrid, const dim3 FvBlock)
{
    return RA::BindStr(
            "(", FvGrid.x,  ',', FvGrid.y,  ',', FvGrid.z,  ")",
            "(", FvBlock.x, ',', FvBlock.y, ',', FvBlock.z, ")");
}