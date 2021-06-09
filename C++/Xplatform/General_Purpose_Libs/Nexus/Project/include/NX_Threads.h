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
    static const int    GetCPUThreadCount();
    static const int    GetThreadCountUsed();
    static const size_t GetInstCount();
    static const size_t GetCrossTaskCount();

    static const int    GetThreadCountAvailable();
    static const bool   ThreadsAreAvailable();

    static void ResetThreadCount();
    static void SetThreadCount(int val);

    void operator+=(int val);
    void operator++();
    void operator-=(int val);
    void operator--();
    void operator==(int val) const;
};

