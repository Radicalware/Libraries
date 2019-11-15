#include "NX_Threads.h"

const int NX_Threads::CPU_THREADS_COUNT = std::thread::hardware_concurrency();
std::atomic<int> NX_Threads::s_Thread_Count = std::thread::hardware_concurrency();
std::atomic<int> NX_Threads::s_Threads_Used = 0;
std::atomic<int> NX_Threads::s_Inst_Count = 0;
std::atomic<int> NX_Threads::s_task_count = 0;

const int NX_Threads::Thread_Count(){
    return NX_Threads::s_Thread_Count;
}

const int NX_Threads::Threads_Used(){
    return s_Threads_Used.load();
}

const size_t NX_Threads::Inst_Count(){
    return NX_Threads::s_Inst_Count.load();
}

const size_t NX_Threads::Cross_Task_Count(){
    return NX_Threads::s_task_count.load();
}
// --------------------------------------------------

const int NX_Threads::Thread_Count_Available(){
    return s_Thread_Count - s_Threads_Used;
}


const bool NX_Threads::Threads_Are_Available(){
    return s_Thread_Count > s_Threads_Used;
}

// --------------------------------------------------

void NX_Threads::Reset_Thread_Count()
{
    s_Thread_Count = CPU_THREADS_COUNT;
}

void NX_Threads::Set_Thread_Count(int val)
{
    if(val > 0)
        s_Thread_Count = val;
}

void NX_Threads::operator+=(int val)
{
    if ((s_Thread_Count + val) < 0)
        s_Thread_Count = 0;
    else
        s_Thread_Count += val;
}

void NX_Threads::operator++()
{
    if ((s_Thread_Count + 1) < 0) 
        s_Thread_Count = 0;
    else
        s_Thread_Count++;
}

void NX_Threads::operator-=(int val)
{
    if ((s_Thread_Count - val) < 0)
        s_Thread_Count = 0;
    else
        s_Thread_Count -= val;
}

void NX_Threads::operator--()
{

    if ((s_Thread_Count - 1) < 1) 
        s_Thread_Count = 0;
    else
        s_Thread_Count--;
}

void NX_Threads::operator==(int val) const
{
    s_Thread_Count = val;
}
