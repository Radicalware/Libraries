#pragma once

#ifndef __CUDA_DEVICE_FUNCTIONS__
#define __CUDA_DEVICE_FUNCTIONS__

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>

#ifndef _uint_
#define _uint_
using uint = size_t;
#endif

#define FenceAndSync() \
    __threadfence(); \
    __syncthreads();

namespace RA
{
    namespace Device
    {
        __device__ uint GetThreadID(const bool Print = false);
        __device__ void Copy(uint3& Left, const uint3& Right);
    };
};

// =====================================================================================================================

__device__ constexpr bool operator==(const uint3& Left, const uint3& Right)
{
    return Left.x == Right.x
        && Left.y == Right.y
        && Left.z == Right.z;
}
__device__ constexpr bool operator!=(const uint3& Left, const uint3& Right)
{
    return !(Left == Right);
}
__device__ void RA::Device::Copy(uint3& Left, const uint3& Right)
{
    Left.x = Right.x;
    Left.y = Right.y;
    Left.z = Right.z;
}


__host__ __device__ void Print(const uint3& LvVertex)
{
    printf("%u.%u.%u\n", LvVertex.x, LvVertex.y, LvVertex.z);
}


template<typename ...A>
__host__ __device__ void Print(const char* FsText, A&&... Args)
{
    printf(FsText, Args...);
}

template<>
__host__ __device__ void Print(const char* FsText)
{
    printf(FsText);
}

// =====================================================================================================================

__device__ uint RA::Device::GetThreadID(const bool Print)
{
    ////First section locates and calculates thread offset within a block
    //const unsigned int Column = threadIdx.x;
    //const unsigned int Row = threadIdx.y;
    //const unsigned int Aisle = threadIdx.z;
    //const unsigned int ThreadsPerRow = blockDim.x; //# threads in x direction aka Row
    //const unsigned int ThreadsPerAisle = (blockDim.x * blockDim.y); //# threads in x and y direction for total threads per Aisle

    //const unsigned int ThreadsPerBlock = (blockDim.x * blockDim.y * blockDim.z);
    //const unsigned int RowOffset = (Row * ThreadsPerRow); //how many rows to push out offset by
    //const unsigned int AisleOffset = (Aisle * ThreadsPerAisle);// how many aisles to push out offset by

    ////Second section locates and caculates block offset withing the grid
    //const unsigned int BlockColumn = blockIdx.x;
    //const unsigned int BlockRow = blockIdx.y;
    //const unsigned int BlockAisle = blockIdx.z;
    //const unsigned int BlocksPerRow = gridDim.x;//# blocks in x direction aka blocks per Row
    //const unsigned int BlocksPerAisle = (gridDim.x * gridDim.y); // # blocks in x and y direction for total blocks per Aisle
    //const unsigned int BlockRowOffset = (BlockRow * BlocksPerRow);// how many rows to push out block offset by
    //const unsigned int BlockAisleOffset = (BlockAisle * BlocksPerAisle);// how many aisles to push out block offset by
    //const unsigned int BlockID = BlockColumn + BlockRowOffset + BlockAisleOffset;

    //const unsigned int BlockOffset = (BlockID * ThreadsPerBlock);

    //const unsigned int GID = (BlockOffset + AisleOffset + RowOffset + Column);

    // https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf

    int LnBlockId = blockIdx.x + blockIdx.y * gridDim.x
                    + gridDim.x * gridDim.y * blockIdx.z;

    int GID = LnBlockId * (blockDim.x * blockDim.y * blockDim.z)
                    + (threadIdx.z * (blockDim.x * blockDim.y))
                    + (threadIdx.y * blockDim.x) + threadIdx.x;


    if (Print)
    {
        printf("blockIdx : (%d,%d,%d) ThreadIdx :(%d,%d,%d), GID : (%2d), input[GID] \n",// :(%2d) \n",
            blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, GID);//, input[gidl]);
    }
    return GID;
}

#endif