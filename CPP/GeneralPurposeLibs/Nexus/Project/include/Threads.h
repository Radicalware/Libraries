#pragma once

#include<thread>
#include<atomic>

#include "RawMapping.h"

namespace RA
{
    class Threads
    {
    protected:
        istatic const int         CPUThreads = std::thread::hardware_concurrency();
        istatic std::atomic<int>  Allowed    = std::thread::hardware_concurrency();
        istatic std::atomic<int>  Used = 0;
        istatic std::atomic<int>  InstanceCount = 0;
        istatic std::atomic<int>  TotalTasksCounted = 0;

    public:
        istatic int  GetCPUThreadCount()       { return RA::Threads::CPUThreads; }
        istatic int  GetAllowedThreadCount()   { return RA::Threads::Allowed; }
        istatic int  GetUsedThreadCount()      { return Used.load(); }
        istatic void IncInstanceCount()        { InstanceCount++; }
        istatic xint GetInstCount()            { return InstanceCount; }
        istatic xint GetTotalTasksRequested()  { return RA::Threads::TotalTasksCounted.load(); }

        istatic int  GetThreadCountAvailable() { return Allowed - Used; }
        istatic bool BxThreadsAreAvailable()   { return Allowed > Used; }

        istatic void ResetAllowedThreadCount() { Allowed = CPUThreads;  }
         static void SetAllowedThreadCount(int FInt);
         
        RIN void operator++() { ++Used;}
        RIN void operator--() { --Used; }
        RIN void operator++(int) { ++Used; }
        RIN void operator--(int) { --Used; }
        RIN void operator==(int FInt) const { Used = FInt; }
    };
};
