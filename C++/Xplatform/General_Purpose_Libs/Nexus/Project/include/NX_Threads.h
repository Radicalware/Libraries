#pragma once

#include<thread>
#include<atomic>



class NX_Threads
{
protected:
    static const int CPU_THREADS_COUNT;
    static std::atomic<int>  s_Thread_Count;
    static std::atomic<int>  s_Threads_Used;
    static std::atomic<int>  s_Inst_Count;
    static std::atomic<int>  s_task_count;

public:
    static int    GetCPUThreadCount();
    static int    GetThreadCountUsed();
    static size_t GetInstCount();
    static size_t GetCrossTaskCount();

    static int    GetThreadCountAvailable();
    static bool   ThreadsAreAvailable();

    static void ResetThreadCount();
    static void SetThreadCount(int val);

    void operator+=(int val);
    void operator++();
    void operator-=(int val);
    void operator--();
    void operator==(int val) const;
};

