#pragma once

#include "BasicCUDA.cuh"

#include <time.h>
#include <math.h>

#ifndef __DEVICE_TIMER__
#define __DEVICE_TIMER__

namespace RA
{
    namespace Device
    {
        class Timer
        {
            clock_t MoStart = abs(clock());

        public:
            __device__ __host__ Timer() {}
            __device__ __host__ void Reset();
            __device__ __host__ xint GetElapsedTimeSeconds() const;
            __device__ __host__ void PrintElapsedTimeSeconds(const char* FsNote = "Time = ") const;
            static __device__ __host__ xint SleepTicks(const xint FnTicks);

        };

        static __device__ __host__ xint SleepTicks(const xint FnTicks);
    }
};

#endif