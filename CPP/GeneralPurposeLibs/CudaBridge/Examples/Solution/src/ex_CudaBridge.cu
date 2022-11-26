
// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>

#ifndef UsingNVCC
#define UsingNVCC
#endif

#pragma warning (disable : 6011 6387)

#include "Macros.h"
#include "Tests.cuh"

#ifdef UsingVLD
#include <vld.h>
#endif

/// Be sure to COPY the following
/// FROM:   "CUDA C/C++"        >> "Additional Include Directories" 
/// TO:     "VC++ Directories"  >> Include Directories


int main()
{
    Begin();
    Nexus<>::Start();

    const uint LnOperations = 1 << 28;
    //const uint LnOperations = 1 << 15; // best for testing mutex (big enough for sample size) (small enough we don't get multi max nums)
    //const uint LnOperations = 1 << 10;

    constexpr auto LnThreadsPerBlock = 1024;
    constexpr auto LnThreadsPerWarps = 32;

    //constexpr auto LnOperations = 15;
    //constexpr auto LnOperations = LnThreadsPerBlock / 15; // 1D
    //constexpr auto LnOperations = LnThreadsPerBlock; // 1D
    //constexpr auto LnOperations = LnThreadsPerBlock + 1; // 1D
    //constexpr auto LnOperations = LnThreadsPerBlock * (LnThreadsPerWarps / 16); // 2D
    //constexpr auto LnOperations = LnThreadsPerBlock * (LnThreadsPerWarps / 8); // 3D
    //constexpr auto LnOperations = 32769;

    cout << "Operations: " << RA::FormatNum(LnOperations) << endl;

    int TestID = 4;
    switch (TestID)
    {
    case 0: break;
    case 1: Test::PrintDeviceStats(); break;
    case 2: Test::PrintGridBlockThread(); break;
    case 3: Test::SumArrayIndicies(); break;
    case 4: Test::TestBlockMutex(LnOperations); 
    case 5: Test::TestThreadMutex(LnOperations); break;
    case 6: Test::TestPrintNestedData(); break;
    default: ThrowIt("Invalid Idx = ", TestID);
    }

    RescuePrint();
    Nexus<>::Stop();
    return 0;
}
