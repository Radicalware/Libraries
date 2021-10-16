#pragma once

#include<thread>
#include<atomic>

namespace RA
{
    class Threads
    {
    protected:
        inline static const int TotalCount = std::thread::hardware_concurrency();
        inline static std::atomic<int>  Remaining = std::thread::hardware_concurrency();
        inline static std::atomic<int>  Used = 0;
        inline static std::atomic<int>  InstanceCount = 0;
        inline static std::atomic<int>  TaskCount = 0;

    public:
        static int      GetCPUThreadCount();
        static int      GetThreadCountUsed();
        static size_t   GetInstCount();
        static size_t   GetCrossTaskCount();

        static int      GetThreadCountAvailable();
        static bool     ThreadsAreAvailable();

        static void     ResetThreadCount();
        static void     SetThreadCount(int val);

        void operator+=(int val);
        void operator++();
        void operator-=(int val);
        void operator--();
        void operator==(int val) const;
    };
};
