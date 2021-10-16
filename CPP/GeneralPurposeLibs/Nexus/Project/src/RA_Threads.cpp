#include "RA_Threads.h"

int RA::Threads::GetCPUThreadCount(){
    return RA::Threads::Remaining;
}

int RA::Threads::GetThreadCountUsed(){
    return Used.load();
}

size_t RA::Threads::GetInstCount(){
    return RA::Threads::InstanceCount.load();
}

size_t RA::Threads::GetCrossTaskCount(){
    return RA::Threads::TaskCount.load();
}
// --------------------------------------------------

int RA::Threads::GetThreadCountAvailable(){
    return Remaining - Used;
}


bool RA::Threads::ThreadsAreAvailable(){
    return Remaining > Used;
}

// --------------------------------------------------

void RA::Threads::ResetThreadCount()
{
    Remaining = TotalCount;
}

void RA::Threads::SetThreadCount(int val)
{
    if(val > 0)
        Remaining = val;
}

void RA::Threads::operator+=(int val)
{
    if ((Remaining + val) < 0)
        Remaining = 0;
    else
        Remaining += val;
}

void RA::Threads::operator++()
{
    if ((Remaining + 1) < 0) 
        Remaining = 0;
    else
        Remaining++;
}

void RA::Threads::operator-=(int val)
{
    if ((Remaining - val) < 0)
        Remaining = 0;
    else
        Remaining -= val;
}

void RA::Threads::operator--()
{

    if ((Remaining - 1) < 1) 
        Remaining = 0;
    else
        Remaining--;
}

void RA::Threads::operator==(int val) const
{
    Remaining = val;
}
