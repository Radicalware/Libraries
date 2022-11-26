#pragma once

#ifndef _DEVICE_MUTEX_
#define _DEVICE_MUTEX_

#include "CudaImport.cuh"
#include "Device.cuh"
#include "Host.cuh"
#include "Timer.cuh"
#include "RawMapping.h"

#include <iostream>

using std::cout;
using std::endl;

#ifndef CudaSwapCMP
#define CudaSwapOffON(_MoMutex_) atomicCAS(_MoMutex_, 0, 1) == 0
#endif



namespace RA
{
    namespace Device
    {
        struct Mutex
        {
            int   MoMutex;
            uint3 MvBlockLock;
            uint3 MvMaxIntBlockLock;
            int*  MvMutexes = nullptr;
            int*  MvRunning = nullptr;
            inline static int SnMtxDepth = 8;

            //int* MvMutexes = nullptr;

            static void ObjInitialize(Mutex& Target, const uint3 FvBlockDim)
            {
                Target.MoMutex = 0;
                Target.MvBlockLock = dim3(MAX_INT, MAX_INT, MAX_INT);
                Target.MvMaxIntBlockLock = dim3(MAX_INT, MAX_INT, MAX_INT);

                Target.MvMutexes = RA::Host::AllocateMemOnDevice<int>(SnMtxDepth + 1);
                Target.MvRunning = RA::Host::AllocateMemOnDevice<int>(SnMtxDepth + 1);
            }

            Mutex(){}

            static void ObjDestroy(Mutex& Target)
            {
                cudaFree(Target.MvMutexes);
                cudaFree(Target.MvRunning);
            }

            ~Mutex()
            {
                Mutex::ObjDestroy(This);
            }

            __device__ bool BxLocked() const { return MoMutex == 1; }

            __device__ void BlockLock() 
            {
                while ((MoMutex > 0 || MvBlockLock == MvMaxIntBlockLock) && MvBlockLock != blockIdx)
                {
                    __syncthreads();
                    if (MvBlockLock == MvMaxIntBlockLock && atomicCAS(&MoMutex, 0, 1) == 0)
                        MvBlockLock = blockIdx;
                }

            }

            __device__ bool BxRunning() const
            {
                for (int i = 0; i < SnMtxDepth; i++)
                    if (MvRunning[i] == false)
                        return false;
                return true;
            }

            __device__ void Unlock()
            {
                __threadfence();
                __syncthreads();
                if (MvBlockLock != MvMaxIntBlockLock)
                {
                    MvBlockLock = MvMaxIntBlockLock;
                    atomicExch(&MoMutex, 0); // set main mutex off
                }
                else
                {
                    for (int i = SnMtxDepth - 1; i >= 0; i--)
                        atomicExch(&MvMutexes[i], 0);
                }
                __threadfence();
                __syncthreads();
            }

            __device__ bool GetThreadLock()
            {
                __syncthreads();
                for (int i = 0; i < SnMtxDepth; i++)
                    MvRunning[i] = !(atomicCAS(&MvMutexes[i], 0, 1) == 0);
                return This.BxRunning();
                __syncthreads();
            }
        };
    }
};

#endif