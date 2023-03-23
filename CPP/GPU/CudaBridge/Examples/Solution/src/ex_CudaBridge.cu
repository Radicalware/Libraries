
// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>

#ifndef UsingNVCC
#define UsingNVCC
#endif

#pragma warning (disable : 6011 6387)

#include "Timer.h"
#include "Macros.h"
#include "Tests.cuh"
#include "Host.cuh"

//#ifndef NDEBUG
//#include "vld.h"
//#endif

/// Be sure to COPY the following
/// FROM:   "CUDA C/C++"        >> "Additional Include Directories" 
/// TO:     "VC++ Directories"  >> Include Directories

template<typename R, typename V>
double NRoot(R FnRoot, V FnVal) 
{
    if (FnVal == 0)
        return 0;
    if(FnVal < 0)
        return -std::pow((-1 * FnVal), (R)1 / FnRoot);
    return std::pow(FnVal, (R)1 / FnRoot);
}

int main()
{
    // std::pow(n, 1/3)
    double x = 5;
    double LnRoot = 3;
    double LnExp = 4;
    double Start = NRoot(LnRoot, pow(x, LnExp));
    cout << Start << endl;

    auto LnBalance1 = (LnExp / LnRoot) + 1;
    auto LnBalance2 = (LnExp / LnRoot) + (LnRoot / LnRoot);

    auto LnNumerator = LnExp + LnRoot;
    auto LnDenom     = LnRoot;
    auto LnBalance3 = (LnNumerator / LnDenom);
    auto LnCrossOut = (LnDenom / LnNumerator);

    auto Mid = (LnCrossOut, pow(x, LnBalance3));
    cout << Mid << endl;

    auto End = LnCrossOut * NRoot(LnDenom, pow(x, LnNumerator));
    cout << End << endl;


    return 0;

    Begin();
    Nexus<>::Start();
    RA::Timer Time;
    //const xint LnOperations = 1 << 28;
    //const xint LnOperations = 1 << 20;
    //const xint LnOperations = 1 << 15; // best for testing mutex (big enough for sample size) (small enough we don't get multi max nums)
    //const xint LnOperations = 1 << 10;
    //const xint LnOperations = 4 * 32;
    //const xint LnOperations = 11;

    constexpr auto LnThreadsPerBlock = 1024;
    constexpr auto LnThreadsPerWarps = 32;

    //constexpr auto LnOperations = 15;
    constexpr auto LnOperations = LnThreadsPerBlock / 15; // 1D
    //constexpr auto LnOperations = LnThreadsPerBlock; // 1D
    //constexpr auto LnOperations = LnThreadsPerBlock + 1; // 1D
    //constexpr auto LnOperations = LnThreadsPerBlock * (LnThreadsPerWarps / 16); // 2D
    //constexpr auto LnOperations = LnThreadsPerBlock * (LnThreadsPerWarps / 8); // 3D
    //constexpr auto LnOperations = 32769;

    cout << "Operations: " << RA::FormatNum(LnOperations) << endl;


    int TestID = 5;
    switch (TestID)
    {
    case 0: break;
    case 1: Test::PrintDeviceStats(); break;
    case 2: Test::Features(); break;
    case 3: Test::PrintGridBlockThread(); break;
    case 4: Test::SumArrayIndiciesMultiStream(LnOperations); break;
    case 5: Test::SumArrayIndiciesMultiGPU(LnOperations); break;
    case 6: Test::TestBlockMutex(LnOperations); break;
    default: ThrowIt("Invalid Idx = ", TestID);
    }
    cout << "Total Execution Time: " << Time.GetElapsedTimeMilliseconds() << endl;
    FinalRescue();
    Nexus<>::Stop();
    return 0;
}
