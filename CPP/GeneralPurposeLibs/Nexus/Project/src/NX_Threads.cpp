#include "NX_Threads.h"

const int NX_Threads::CPU_THREADS_COUNT = std::thread::hardware_concurrency();
std::atomic<int> NX_Threads::s_Thread_Count = std::thread::hardware_concurrency();
std::atomic<int> NX_Threads::s_Threads_Used = 0;
std::atomic<int> NX_Threads::s_Inst_Count = 0;
std::atomic<int> NX_Threads::s_task_count = 0;

int NX_Threads::GetCPUThreadCount(){
    return NX_Threads::s_Thread_Count;
}

int NX_Threads::GetThreadCountUsed(){
    return s_Threads_Used.load();
}

size_t NX_Threads::GetInstCount(){
    return NX_Threads::s_Inst_Count.load();
}

size_t NX_Threads::GetCrossTaskCount(){
    return NX_Threads::s_task_count.load();
}
// --------------------------------------------------

int NX_Threads::GetThreadCountAvailable(){
    return s_Thread_Count - s_Threads_Used;
}


bool NX_Threads::ThreadsAreAvailable(){
    return s_Thread_Count > s_Threads_Used;
}

// --------------------------------------------------

void NX_Threads::ResetThreadCount()
{
    s_Thread_Count = CPU_THREADS_COUNT;
}

void NX_Threads::SetThreadCount(int val)
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
