#include "Threads.h"

void RA::Threads::SetAllowedThreadCount(int FInt)
{
    if(FInt > 0)
        Allowed = MIN(FInt, CPUThreads);
        
}

