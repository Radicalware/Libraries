#include "Threads.h"


int RA::Threads::GetCPUThreadCount() {
    return RA::Threads::CPUThreads;
}

int RA::Threads::GetAllowedThreadCount(){
    return RA::Threads::Allowed;
}

int RA::Threads::GetThreadCountUsed(){
    return Used.load();
}

xint RA::Threads::GetInstCount(){
    return RA::Threads::InstanceCount.load();
}

xint RA::Threads::GetTotalTasksRequested(){
    return RA::Threads::TotalTasksCounted.load();
}
// --------------------------------------------------

int RA::Threads::GetThreadCountAvailable(){
    return Allowed - Used;
}


bool RA::Threads::ThreadsAreAvailable(){
    return Allowed > Used;
}

// --------------------------------------------------

void RA::Threads::ResetAllowedThreadCount()
{
    Allowed = CPUThreads;
}

void RA::Threads::SetAllowedThreadCount(int val)
{
    if(val > 0)
        Allowed = val;
}

void RA::Threads::operator+=(int val)
{
    if ((Allowed + val) < 0)
        Allowed = 0;
    else
        Allowed += val;
}

void RA::Threads::operator++()
{
    if ((Allowed + 1) < 0) 
        Allowed = 0;
    else
        Allowed++;
}

void RA::Threads::operator-=(int val)
{
    if ((Allowed - val) < 0)
        Allowed = 0;
    else
        Allowed -= val;
}

void RA::Threads::operator--()
{

    if ((Allowed - 1) < 1) 
        Allowed = 0;
    else
        Allowed--;
}

void RA::Threads::operator==(int val) const
{
    Allowed = val;
}
