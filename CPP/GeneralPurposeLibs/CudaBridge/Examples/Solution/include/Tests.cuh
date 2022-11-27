#pragma once

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include "Macros.h"

namespace Test
{
    void PrintDeviceStats();
    void PrintGridBlockThread();
    void SumArrayIndicies();
    void TestBlockMutex(const uint FnOperations);
}