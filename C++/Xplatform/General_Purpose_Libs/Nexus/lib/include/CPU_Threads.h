#pragma once

#include<thread>

class CPU_Threads
{
protected:
	static int     Thread_Count;
	static int     Threads_Used;

	static size_t  Inst_Count;
	static size_t  Task_Count;

public:
	static const size_t thread_count();
	static const int    threads_available();
	static const int    threads_used();
	static const bool   threads_are_available();

	static const size_t inst_count();
	static const size_t cross_task_count();
};


