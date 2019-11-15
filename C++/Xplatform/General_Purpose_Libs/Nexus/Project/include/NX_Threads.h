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
    static const int    Thread_Count();
    static const int    Threads_Used();
    static const size_t Inst_Count();
    static const size_t Cross_Task_Count();

    static const int    Thread_Count_Available();
    static const bool   Threads_Are_Available();

    static void Reset_Thread_Count();
    static void Set_Thread_Count(int val);

    void operator+=(int val);
    void operator++();
    void operator-=(int val);
    void operator--();
    void operator==(int val) const;
};

