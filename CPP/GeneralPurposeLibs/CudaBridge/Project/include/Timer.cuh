#pragma once

#include "CudaImport.cuh"

#include <time.h>

#ifndef __DEVICE_TIMER__
#define __DEVICE_TIMER__

namespace RA
{
    namespace Device
    {
        class Timer
        {
            clock_t MoStart = clock();

        public:
            __device__ Timer() {}
            __device__ void Reset();
            __device__ uint GetElapsedTimeSeconds() const;
            __device__ void PrintElapsedTimeSeconds(const char* FsNote = "Time = ") const;

            __device__ void SleepTicks(const uint FnTicks) const;
        };
    }
};


__device__ void RA::Device::Timer::Reset()
{
    clock_t MoStart = clock();
}

__device__ uint RA::Device::Timer::GetElapsedTimeSeconds() const
{
    clock_t LoFinish = clock();
    return static_cast<uint>(LoFinish - MoStart);
}

__device__ void RA::Device::Timer::PrintElapsedTimeSeconds(const char* FsNote) const
{
    printf("%s%llu\n", FsNote, GetElapsedTimeSeconds());
}

__device__ void RA::Device::Timer::SleepTicks(const uint FnTicks) const
{
    while ((static_cast<uint>(clock() - MoStart)) < FnTicks);
}



#endif