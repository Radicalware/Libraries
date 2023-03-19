#pragma once

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include "Macros.h"

namespace Test
{
    void PrintDeviceStats();
    void Features();
    void PrintGridBlockThread();
    void SumArrayIndiciesMultiStream(const xint FnOperations);
    void SumArrayIndiciesMultiGPU(const xint FnOperations);
    void TestBlockMutex(const xint FnOperations);
}