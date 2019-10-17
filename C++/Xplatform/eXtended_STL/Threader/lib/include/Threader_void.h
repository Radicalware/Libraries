#pragma once

#include<iostream>
#include<initializer_list>
#include<utility>

#include<future>
#include<thread>
#include<mutex>
#include<condition_variable>

#include<deque>
#include<functional>
#include<type_traits>

#include "Threader_T.h"
#include "CPU_Threads.h"
#include "Task.h"
#include "Job.h"

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
#include<Windows.h>
#else 
#include<unistd.h>
#endif


#include<iostream>
#include<initializer_list>
#include<utility>

#include<future>
#include<thread>
#include<mutex>
#include<condition_variable>

#include<queue>
#include<functional>
#include<type_traits>

#include "Threader_T.h"
#include "CPU_Threads.h"
#include "Task.h"

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
#include<Windows.h>
#else 
#include<unistd.h>
#endif



// =========================================================================================

template<>
class Threader<void> : public CPU_Threads
{
private:
	bool m_finish_tasks = false;
	size_t m_inst_task_count = 0;
	std::mutex m_mutex;
	std::condition_variable m_sig_deque;
	std::vector<std::thread> m_threads;      // We add new tasks by succession from the m_task_queue here
	std::queue<Task<void>> m_task_queue; // This is where tasks are held before being run

	void TaskLooper(int thread_idx);

public:
	Threader();
	~Threader();

	template <typename F, typename... A>
	void add_job(F&& function, A&& ... args);

	size_t size() const;

	void wait_all() const;
	void clear();

	void sleep(unsigned int extent) const;
};

// =========================================================================================

inline void Threader<void>::TaskLooper(int thread_idx)
{
	while (true) {
		Task<void>* tsk = nullptr;
		{
			std::unique_lock<std::mutex> lock(m_mutex);
			m_sig_deque.wait(lock, [this]() {
				return ((m_finish_tasks || m_task_queue.size()) && CPU_Threads::threads_are_available());
			});

			if (m_task_queue.empty())
				return;

			CPU_Threads::Threads_Used++;
			tsk = new Task<void>(std::move(m_task_queue.front()));

			CPU_Threads::Task_Count++;
			m_inst_task_count++;

			m_task_queue.pop();
		}
		(*tsk)();
		CPU_Threads::Threads_Used--; // protected as atomic
		if(tsk != nullptr)
			delete tsk;
	}
}


inline Threader<void>::Threader()
{
	CPU_Threads::Inst_Count++;
	m_threads.reserve(CPU_Threads::Thread_Count);
	for (int i = 0; i < CPU_Threads::Thread_Count; ++i)
		m_threads.emplace_back(std::bind(&Threader<void>::TaskLooper, this, i));

	CPU_Threads::Threads_Used = 0;
}

inline Threader<void>::~Threader()
{
	{
		std::unique_lock <std::mutex> lock(m_mutex);
		m_finish_tasks = true;
		m_sig_deque.notify_all();
	}
    for (auto& thrd : m_threads) thrd.join();
}

template <typename F, typename... A>
inline void Threader<void>::add_job(F&& function, A&&... args) 
{
	auto binded_function = std::bind(std::forward<F>(function), std::forward<A>(args)...);
	std::lock_guard <std::mutex> lock(m_mutex);
	m_task_queue.emplace(std::move(binded_function));
	m_sig_deque.notify_one();
}

inline size_t Threader<void>::size() const{
	return m_inst_task_count;
}

inline void Threader<void>::wait_all() const 
{
	while (m_task_queue.size()) this->sleep(1);
	while (CPU_Threads::Threads_Used > 0) this->sleep(1);
}

inline void Threader<void>::clear() {
	m_inst_task_count = 0;
}

inline void Threader<void>::sleep(unsigned int extent) const
{
	#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
		::Sleep(extent);
	#else
		::usleep(extent);
	#endif
}
