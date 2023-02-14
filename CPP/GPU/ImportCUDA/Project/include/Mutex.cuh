#pragma once

#ifndef _DEVICE_MUTEX_
#define _DEVICE_MUTEX_

#include "BasicCUDA.cuh"
#include "Device.cuh"
#include "Host.cuh"
#include "Timer.cuh"
#include "RawMapping.h"

#include <iostream>

using std::cout;
using std::endl;

#ifndef CudaSwapCMP
#define CudaSwapOffON(_MoBlockMutex_) atomicCAS(_MoBlockMutex_, 0, 1) == 0
#endif

namespace RA
{
    namespace Device
    {
        class Mutex
        {
            int   MoBlockMutex;
            uint3 MvBlockLock;
            uint3 MvMaxIntBlockLock;

        public:
            void Configure(Mutex& Target, const uint3 FvGrid)
            {
                Target.MoBlockMutex = 0;
                Target.MvBlockLock = dim3(MAX_INT, MAX_INT, MAX_INT);
                Target.MvMaxIntBlockLock = dim3(MAX_INT, MAX_INT, MAX_INT);
            }

            Mutex() {}

            // ----------------------------------------------------------------
            // All Mutex Fucntions
            __device__ void ResetMutex()
            {
                FenceAndSync();
                MvBlockLock = MvMaxIntBlockLock;
                atomicExch(&MoBlockMutex, 0);
                FenceAndSync();
            }
            // ----------------------------------------------------------------
            // Block Mutex Fucntions (Fast but Atomics Needed)
            __device__ bool BxBlockLocked() const { return MoBlockMutex == 1; }

            __device__ void BlockLock()
            {
                __syncthreads();
                while ((MoBlockMutex > 0 || MvBlockLock == MvMaxIntBlockLock) && MvBlockLock != blockIdx)
                {
                    __syncthreads();
                    if (MvBlockLock == MvMaxIntBlockLock && atomicCAS(&MoBlockMutex, 0, 1) == 0)
                        MvBlockLock = blockIdx;
                }
                __syncthreads();
            }

            __device__ void UnlockBlocks()
            {
                FenceAndSync();
                MvBlockLock = MvMaxIntBlockLock;
                atomicExch(&MoBlockMutex, 0); // set main mutex off
                FenceAndSync();
            }
        };
    }
};

#endif