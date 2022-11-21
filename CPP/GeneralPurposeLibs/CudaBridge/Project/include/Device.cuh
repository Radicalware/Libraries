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


__device__ void Print(const uint3& LvVertex)
{
    printf("%u.%u.%u", LvVertex.x, LvVertex.y, LvVertex.z);
}


template<typename ...A>
__device__ void Print(const char* FsText, A&&... Args)
{
    printf(FsText, Args...);
}

// =====================================================================================================================

__device__ uint RA::Device::GetThreadID(const bool Print)
{
    //First section locates and calculates thread offset within a block
    int Column = threadIdx.x;
    int Row = threadIdx.y;
    int Aisle = threadIdx.z;
    int ThreadsPerRow = blockDim.x; //# threads in x direction aka Row
    int ThreadsPerAisle = (blockDim.x * blockDim.y); //# threads in x and y direction for total threads per Aisle

    int ThreadsPerBlock = (blockDim.x * blockDim.y * blockDim.z);
    int RowOffset = (Row * ThreadsPerRow); //how many rows to push out offset by
    int AisleOffset = (Aisle * ThreadsPerAisle);// how many aisles to push out offset by

    //Second section locates and caculates block offset withing the grid
    int BlockColumn = blockIdx.x;
    int BlockRow = blockIdx.y;
    int BlockAisle = blockIdx.z;
    int BlocksPerRow = gridDim.x;//# blocks in x direction aka blocks per Row
    int BlocksPerAisle = (gridDim.x * gridDim.y); // # blocks in x and y direction for total blocks per Aisle
    int BlockRowOffset = (BlockRow * BlocksPerRow);// how many rows to push out block offset by
    int BlockAisleOffset = (BlockAisle * BlocksPerAisle);// how many aisles to push out block offset by
    int BlockID = BlockColumn + BlockRowOffset + BlockAisleOffset;

    int BlockOffset = (BlockID * ThreadsPerBlock);

    int GID = (BlockOffset + AisleOffset + RowOffset + Column);

    if (Print)
    {
        printf("blockIdx : (%d,%d,%d) ThreadIdx :(%d,%d,%d), GID : (%2d), input[GID] \n",// :(%2d) \n",
            blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, GID);//, input[gidl]);
    }
    return GID;
}

#endif