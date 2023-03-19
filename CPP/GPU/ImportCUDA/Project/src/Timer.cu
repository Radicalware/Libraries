#include "Timer.cuh"

__device__ __host__ void RA::Device::Timer::Reset()
{
    clock_t MoStart = abs(clock());
}

__device__ __host__ xint RA::Device::Timer::GetElapsedTimeSeconds() const
{
    clock_t LoFinish = abs(clock());
    return static_cast<xint>(LoFinish - MoStart);
}

__device__ __host__ void RA::Device::Timer::PrintElapsedTimeSeconds(const char* FsNote) const
{
    printf("%s%llu\n", FsNote, GetElapsedTimeSeconds());
}

__device__ __host__ xint RA::Device::Timer::SleepTicks(const xint FnTicks)
{
    return RA::Device::SleepTicks(FnTicks);
}

__device__ __host__ xint RA::Device::SleepTicks(const xint FnTicks)
{
    //auto FnFuture = abs(clock()) + FnTicks;
    //while ((static_cast<xint>(FnFuture)) > abs(clock()));
    //xint LnCount = 0;
    ////while (LnCount < FnTicks)
    ////    LnCount++;
    //return LnCount;

    xint n, t1 = 0, t2 = 1, nextTerm = 0;
    n = FnTicks;
    for (xint i = 1; i <= n; ++i) {
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