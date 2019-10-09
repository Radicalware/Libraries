#pragma once

#include<thread>
#include<atomic>

class CPU_Threads
{
protected:
	const  int              CPU_THREADS = std::thread::hardware_concurrency();
	static int              Thread_Count;

	static std::atomic<int> Threads_Used;
	static std::atomic<int>  Inst_Count;
	static std::atomic<int>  Task_Count;

public:
	static const int    thread_count();
	static const int    threads_used();
	static const size_t inst_count();
	static const size_t cross_task_count();

	static const int    threads_available();
	static const bool   threads_are_available();

	void reset_thread_count();
	void operator+=(int val);
	void operator-=(int val);
	void operator==(int val) const;
};


