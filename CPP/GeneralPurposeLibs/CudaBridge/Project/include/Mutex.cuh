#pragma once

#ifndef _DEVICE_MUTEX_
#define _DEVICE_MUTEX_

#include "CudaImport.cuh"
#include "Device.cuh"
#include "Host.cuh"

#include <iostream>

using std::cout;
using std::endl;

#ifndef CudaSwapCMP
#define CudaSwapOffON(_MoMutex_) atomicCAS(_MoMutex_, Off, On) == Off
#endif



namespace RA
{
    namespace Device
    {
        struct Mutex
        {
            int   MoMutex = 0;
            bool  MbMutexRunning = false;
            uint3 MvBlockLock     = dim3(MAX_INT, MAX_INT, MAX_INT);
            uint3 MvOpenLockBlock = dim3(MAX_INT, MAX_INT, MAX_INT);

            Mutex() { }

            __device__ __host__ uint static GetBufferSize() 
            { 
                return sizeof(Mutex) + sizeof(uint); 
            }

            __device__ bool BxLocked() { return MoMutex == On; }

            __device__ void BlockLock(const uint3 LvBlock) 
            {
                while ((MoMutex > 0 || MvBlockLock == MvOpenLockBlock) && MvBlockLock != LvBlock)
                {
                    if (MvBlockLock == MvOpenLockBlock && atomicCAS(&MoMutex, Off, On) == Off)
                        MvBlockLock = LvBlock;
                }
            }

            __device__ void Unlock()
            {
                MvBlockLock = MvOpenLockBlock;
                atomicExch(&MoMutex, Off);
            }

            __device__ bool BxGetLock()
            {
                return !(atomicCAS(&MoMutex, 0, 1) == 0);
            }
        };
    }
};

#endif