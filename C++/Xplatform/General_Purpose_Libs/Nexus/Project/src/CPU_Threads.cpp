#include "CPU_Threads.h"

const int CPU_Threads::CPU_THREADS_COUNT = std::thread::hardware_concurrency() - 1;
std::atomic<int> CPU_Threads::Thread_Count = std::thread::hardware_concurrency() - 1;
std::atomic<int> CPU_Threads::Threads_Used = 0;
std::atomic<int> CPU_Threads::Inst_Count = 0;
std::atomic<int> CPU_Threads::Task_Count = 0;

const int CPU_Threads::thread_count(){
    return CPU_Threads::Thread_Count;
}

const int CPU_Threads::threads_used(){
    return Threads_Used.load();
}

const size_t CPU_Threads::inst_count(){
    return CPU_Threads::Inst_Count.load();
}

const size_t CPU_Threads::cross_task_count(){
    return CPU_Threads::Task_Count.load();
}
// --------------------------------------------------

const int CPU_Threads::threads_available(){
    return Thread_Count - Threads_Used;
}


const bool CPU_Threads::threads_are_available(){
    return Thread_Count > Threads_Used;
}

// --------------------------------------------------

void CPU_Threads::reset_thread_count()
{
    Thread_Count = CPU_THREADS_COUNT;
}

void CPU_Threads::operator+=(int val)
{
    if ((Thread_Count + val) < 0)
        Thread_Count = 0;
    else
        Thread_Count += val;
}

void CPU_Threads::operator++()
{
    if ((Thread_Count + 1) < 0) 
        Thread_Count = 0;
    else
        Thread_Count++;
}

void CPU_Threads::operator-=(int val)
{
    if ((Thread_Count - val) < 0)
        Thread_Count = 0;
    else
        Thread_Count -= val;
}

void CPU_Threads::operator--()
{

    if ((Thread_Count - 1) < 1) 
        Thread_Count = 0;
    else
        Thread_Count--;
}

void CPU_Threads::operator==(int val) const
{
    Thread_Count = val;
}

void CPU_Threads::set_thread_count(int val)
{
    if(val > 0)
        Thread_Count = val;
}
