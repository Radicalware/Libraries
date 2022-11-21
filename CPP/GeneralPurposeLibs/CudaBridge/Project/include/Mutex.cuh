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

            Mutex()
            {

            }

            __device__ __host__ uint static GetBufferSize() 
            { 
                return sizeof(Mutex) + sizeof(uint); 
            }

            __device__ bool BxLocked() { return MoMutex == True; }

            __device__ void BlockLock(const uint3 LvBlock) 
            {
                //printf("Starting Lock: %u.%u.%u\n", LvBlock.x, LvBlock.y, LvBlock.z);
                //printf("Current Block: %u.%u.%u\n", MvBlockLock.x, MvBlockLock.y, MvBlockLock.z);
                while ((MoMutex > 0 || MvBlockLock == MvOpenLockBlock) && MvBlockLock != LvBlock)
                {
                    // 1st = *MvBlockLockPtr == *MvOpenLockBlockPtr && *MvBlockLockPtr != LvBlock
                    // 2nd = mutex > 0 && *MvBlockLockPtr != LvBlock
                    // pass if mutex open and block equal
                    if (MvBlockLock == MvOpenLockBlock && atomicCAS(&MoMutex, Off, On) == Off)
                    {
                        MvBlockLock = LvBlock;
                        //Print("New Block Lock: "); Print(LvBlock); Print("\n");
                    }
                }
            }

            __device__ bool BxUnlocked()
            {
                return !(atomicCAS(&MoMutex, 0, 1) == 0);
            }


            __device__ void Unlock()
            {
                MvBlockLock = MvOpenLockBlock;
                atomicExch(&MoMutex, Off);
            }
        };
    }
};

#endif