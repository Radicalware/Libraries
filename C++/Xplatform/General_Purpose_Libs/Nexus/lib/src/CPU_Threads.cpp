#include "CPU_Threads.h"

int     CPU_Threads::Thread_Count = std::thread::hardware_concurrency();
int     CPU_Threads::Threads_Used = 0;
size_t  CPU_Threads::Inst_Count = 0;
size_t  CPU_Threads::Task_Count = 0;

const size_t CPU_Threads::thread_count()
{
	return CPU_Threads::Thread_Count;
}

const int CPU_Threads::threads_available()
{
	return Thread_Count - Threads_Used;
}

const int CPU_Threads::threads_used()
{
	return Threads_Used;
}
#include<iostream>
const bool CPU_Threads::threads_are_available()
{
	return Thread_Count > Threads_Used;
}

const size_t CPU_Threads::inst_count()
{
	return CPU_Threads::Inst_Count;
}

const size_t CPU_Threads::cross_task_count()
{
	return CPU_Threads::Task_Count;
}

