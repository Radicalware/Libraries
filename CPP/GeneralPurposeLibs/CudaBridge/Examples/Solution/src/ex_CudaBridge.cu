
// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>

#ifndef UsingNVCC
#define UsingNVCC
#endif

#include "Macros.h"
#include "Tests.cuh"

/// Be sure to COPY the following
/// FROM:   "CUDA C/C++"        >> "Additional Include Directories" 
/// TO:     "VC++ Directories"  >> Include Directories

int main()
{
    Begin();
    Nexus<>::Start();

    const uint LnOperations = 500;

    int TestID = 4;
    switch (TestID)
    {
    case 0: break;
    case 1: Test::PrintDeviceStats(); break;
    case 2: Test::PrintGridBlockThread(); break;
    case 3: Test::SumArrayIndicies(); break;
    case 4: Test::TestBlockMutex(LnOperations);
    case 5: Test::TestThreadMutex(LnOperations); break;
    default: ThrowIt("Invalid Idx = ", TestID);
    }

    RescuePrint();
    Nexus<>::Stop();
    return 0;
}
