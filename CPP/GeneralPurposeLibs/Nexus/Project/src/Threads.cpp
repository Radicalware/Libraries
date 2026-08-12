#include "Threads.h"

void RA::Threads::SetThreadCount(int FInt)
{
    if (FInt > 0)
        Allowed = MIN(FInt, CPUThreads);
    else
        Allowed = CPUThreads;
}

