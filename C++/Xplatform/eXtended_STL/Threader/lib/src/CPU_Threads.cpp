#include "CPU_Threads.h"

int CPU_Threads::Thread_Count = std::thread::hardware_concurrency();

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
	Thread_Count = CPU_THREADS;
}

void CPU_Threads::operator+=(int val)
{
	Thread_Count += val;
	if (Thread_Count < 0)
		Thread_Count = 0;
}

void CPU_Threads::operator-=(int val)
{
	Thread_Count -= val;
	if (Thread_Count < 0)
		Thread_Count = 0;
}

void CPU_Threads::operator==(int val) const
{
	Thread_Count = val;
}
