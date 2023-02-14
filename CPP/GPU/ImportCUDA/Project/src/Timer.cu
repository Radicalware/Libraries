#include "Timer.cuh"

__device__ __host__ void RA::Device::Timer::Reset()
{
    clock_t MoStart = abs(clock());
}

__device__ __host__ uint RA::Device::Timer::GetElapsedTimeSeconds() const
{
    clock_t LoFinish = abs(clock());
    return static_cast<uint>(LoFinish - MoStart);
}

__device__ __host__ void RA::Device::Timer::PrintElapsedTimeSeconds(const char* FsNote) const
{
    printf("%s%llu\n", FsNote, GetElapsedTimeSeconds());
}

__device__ __host__ uint RA::Device::Timer::SleepTicks(const uint FnTicks)
{
    return RA::Device::SleepTicks(FnTicks);
}

__device__ __host__ uint RA::Device::SleepTicks(const uint FnTicks)
{
    //auto FnFuture = abs(clock()) + FnTicks;
    //while ((static_cast<uint>(FnFuture)) > abs(clock()));
    //uint LnCount = 0;
    ////while (LnCount < FnTicks)
    ////    LnCount++;
    //return LnCount;

    uint n, t1 = 0, t2 = 1, nextTerm = 0;
    n = FnTicks;
    for (uint i = 1; i <= n; ++i) {
        // Prints the first two terms.
        if (i == 1) {
            continue;
        }
        if (i == 2) {
            continue;
        }
        nextTerm = t1 + t2;
        t1 = t2;
        t2 = nextTerm;
    }
    return nextTerm;
}