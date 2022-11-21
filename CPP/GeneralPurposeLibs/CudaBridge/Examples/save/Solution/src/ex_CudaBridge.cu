
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

    int TestID = 2;
    switch (TestID)
    {
    case 0: break;
    case 1: Test::FindMaxIdx(); break;
    case 2: Test::SumArrayIndicies(); break;
    default: ThrowIt("Invalid Idx = ", TestID);
    }

    RescuePrint();
    Nexus<>::Stop();
    return 0;
}
