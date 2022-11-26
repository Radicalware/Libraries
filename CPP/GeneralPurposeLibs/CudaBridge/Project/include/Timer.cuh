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
            __device__ __host__ Timer() {}
            __device__ __host__ void Reset();
            __device__ __host__ uint GetElapsedTimeSeconds() const;
            __device__ __host__ void PrintElapsedTimeSeconds(const char* FsNote = "Time = ") const;
            __device__ __host__ void SleepTicks(const uint FnTicks) const;

        };

        __device__ __host__ void SleepTicks(const uint FnTicks);
    }
};


__device__ __host__ void RA::Device::Timer::Reset()
{
    clock_t MoStart = clock();
}

__device__ __host__ uint RA::Device::Timer::GetElapsedTimeSeconds() const
{
    clock_t LoFinish = clock();
    return static_cast<uint>(LoFinish - MoStart);
}

__device__ __host__ void RA::Device::Timer::PrintElapsedTimeSeconds(const char* FsNote) const
{
    printf("%s%llu\n", FsNote, GetElapsedTimeSeconds());
}

__device__ __host__ void RA::Device::Timer::SleepTicks(const uint FnTicks) const
{
    RA::Device::SleepTicks(FnTicks);
}

__device__ __host__ void RA::Device::SleepTicks(const uint FnTicks)
{
    auto FnFuture = clock() + FnTicks;
    while ((static_cast<uint>(FnFuture)) > clock());
}

#endif