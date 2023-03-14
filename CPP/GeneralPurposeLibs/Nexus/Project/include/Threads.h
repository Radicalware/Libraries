﻿#pragma once

#include<thread>
#include<atomic>

#include "RawMapping.h"

namespace RA
{
    class Threads
    {
    protected:
        inline static const int         CPUThreads = std::thread::hardware_concurrency();
        inline static std::atomic<int>  Allowed    = std::thread::hardware_concurrency();
        inline static std::atomic<int>  Used = 0;
        inline static std::atomic<int>  InstanceCount = 0;
        inline static std::atomic<int>  TotalTasksCounted = 0;

    public:
        static int  GetCPUThreadCount();
        static int  GetAllowedThreadCount();
        static int  GetThreadCountUsed();
        static xint GetInstCount();
        static xint GetTotalTasksRequested();

        static int  GetThreadCountAvailable();
        static bool ThreadsAreAvailable();

        static void ResetAllowedThreadCount();
        static void SetAllowedThreadCount(int val);

        void operator+=(int val);
        void operator++();
        void operator-=(int val);
        void operator--();
        void operator==(int val) const;
    };
};
