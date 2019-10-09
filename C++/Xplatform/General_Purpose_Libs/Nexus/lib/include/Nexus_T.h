
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

#include "xvector.h"
#include "xstring.h"
#include "xmap.h"

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

#include<deque>
#include<functional>
#include<type_traits>

#include "xvector.h"
#include "xstring.h"
#include "xmap.h"

#include "CPU_Threads.h"
#include "Task.h"
#include "Job.h"

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
#include<Windows.h>
#else 
#include<unistd.h>
#endif



// =========================================================================================

namespace util {
	template <typename T>
	using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;
}

template<typename T = int>
class Nexus : public CPU_Threads
{
private:
	bool m_finish_tasks = false;
	size_t m_inst_task_count = 0;
	std::mutex m_mutex;
	std::condition_variable m_sig_deque;
	std::mutex m_get_mutex;
	std::condition_variable m_sig_get;
	xvector<std::thread> m_threads;   // We add new tasks by succession from the m_task_deque here
	std::deque<Task<T>> m_task_deque; // This is where tasks are held before being run
	// deque chosen over queue because the deque has an iterator
	xmap<const xstring*, size_t> m_str_inst_xm; // KVP (xstring >> job inst)
	xmap<size_t, Job<T>>* m_inst_job_xm;        //            KVP (job inst >> Job)
	// m_inst_job_xm uses a map because Job<T> addresses must stay constant (they will change as a vector)
	void TaskLooper(int thread_idx);

	template <typename F, typename... A>
	void add(const xstring& key, F&& function, A&& ... Args);
public:
	Nexus();
	~Nexus();

	template <typename F, typename... A>
	void add_job(const xstring& key, F&& function, A&& ... Args);
	template <typename F, typename... A>
	void add_job(const char* key, F&& function, A&& ... Args);
	template <typename F, typename... A> // the "enable_if_t" restraint was designed by "Ben" from "Cpplang" on Slack!
	auto add_job(F&& function, A&& ... Args)->std::enable_if_t<!std::is_same_v<util::remove_cvref_t<F>, xstring>, void>;

	// Getters can't be const due to the mutex
	Job<T> operator()(const xstring& val);
	Job<T> operator()(const size_t val);
	// Getters can't be const due to the mutex
	Job<T> get(const xstring& val);
	Job<T> get(const size_t val);

	size_t size() const;
	void wait_all() const;
	void clear();
	void sleep(unsigned int extent) const;
};

// =========================================================================================

template<typename T>
inline void Nexus<T>::TaskLooper(int thread_idx)
{
	while (true) {
		size_t tsk_idx;
		{
			std::unique_lock<std::mutex> lock(m_mutex);
			m_sig_deque.wait(lock, [this]() {
				return ((m_finish_tasks || m_task_deque.size()) && CPU_Threads::threads_are_available());
				});

			if (m_task_deque.empty())
				return;

			CPU_Threads::Threads_Used++;
			tsk_idx = m_inst_task_count;
			if (m_task_deque.front().blank())
				continue;
			else
				(*m_inst_job_xm).add_pair(tsk_idx, Job<T>(std::move(m_task_deque.front()), CPU_Threads::Task_Count));

			const Task<T>* latest_task = (*m_inst_job_xm)[tsk_idx].task_ptr();
			if (latest_task->has_name())
				m_str_inst_xm.add_pair(latest_task->name_ptr(), tsk_idx);

			CPU_Threads::Task_Count++;
			m_inst_task_count++;

			m_task_deque.pop_front();
			m_sig_get.notify_all();
		}
		(*m_inst_job_xm)[tsk_idx].init();
	}
}

template<typename T>
template<typename F, typename ...A>
inline void Nexus<T>::add(const xstring& key, F&& function, A&& ...Args)
{
	std::lock_guard<std::mutex> glock(m_get_mutex);
	auto binded_function = std::bind(std::forward<F>(function), std::forward<A>(Args)...);
	std::lock_guard <std::mutex> lock(m_mutex);
	if (key.size())
		m_task_deque.emplace_back(std::move(binded_function), key);
	else
		m_task_deque.emplace_back(std::move(binded_function));
	m_sig_deque.notify_one();
	m_sig_get.notify_all();
}
// ------------------------------------------------------------------------------------------
template<typename T>
Nexus<T>::Nexus()
{
	m_inst_job_xm = new xmap<size_t, Job<T>>;
	CPU_Threads::Inst_Count++;
	m_threads.reserve(CPU_Threads::Thread_Count);
	for (int i = 0; i < CPU_Threads::Thread_Count; ++i)
		m_threads.emplace_back(std::bind(&Nexus<T>::TaskLooper, this, i));

	CPU_Threads::Threads_Used = 0;
}

template<typename T>
Nexus<T>::~Nexus()
{
	{
		std::unique_lock <std::mutex> lock(m_mutex);
		m_finish_tasks = true;
		m_sig_deque.notify_all();
	}
	m_threads.proc([](auto& t) { t.join(); });
	delete m_inst_job_xm;
}
// ------------------------------------------------------------------------------------------
template<typename T>
template<typename F, typename ...A>
inline void Nexus<T>::add_job(const xstring& key, F&& function, A&& ... Args)
{
	this->add(key, std::forward<F>(function), std::forward<A>(Args)...);
}

template<typename T>
template<typename F, typename ...A>
inline void Nexus<T>::add_job(const char* key, F&& function, A&& ...Args)
{
	this->add(xstring(key), std::forward<F>(function), std::forward<A>(Args)...);
}

template<typename T>
template <typename F, typename... A>
inline auto Nexus<T>::add_job(F&& function, A&& ... Args)->
std::enable_if_t<!std::is_same_v<util::remove_cvref_t<F>, xstring>, void>
{
	this->add(xstring(), std::forward<F>(function), std::forward<A>(Args)...);
}
// ------------------------------------------------------------------------------------------

template<typename T>
inline Job<T> Nexus<T>::operator()(const xstring& input) 
{
	std::unique_lock<std::mutex> glock(m_get_mutex);
	if (m_task_deque.size())
		m_sig_get.wait(glock);

	if (!m_str_inst_xm.has(input)) {
		for (const Task<T>& t : m_task_deque) {
			if (*t.name_ptr() == input)
				break;
		}
		throw std::runtime_error("Nexus Key Not Found!");
	}
	while (!m_str_inst_xm.has(input)) this->sleep(1);
	size_t loc = m_str_inst_xm[input];
	while (!(*m_inst_job_xm)[loc].done()) this->sleep(1);

	(*m_inst_job_xm)[loc].rethrow_exception();
	return (*m_inst_job_xm)[loc];
}


template<typename T>
inline Job<T> Nexus<T>::operator()(const size_t input) 
{
	std::unique_lock<std::mutex> glock(m_get_mutex);
	if (m_task_deque.size())
		m_sig_get.wait(glock);

	if (m_inst_task_count + m_task_deque.size() < input + 1)
		throw std::runtime_error("Requested Job is Out of Range\n");

	while (!(*m_inst_job_xm)(input)) this->sleep(1);
	while (!(*m_inst_job_xm)[input].done()) this->sleep(1);

	(*m_inst_job_xm)[input].rethrow_exception();
	return (*m_inst_job_xm)[input];
}

template<typename T>
inline Job<T> Nexus<T>::get(const xstring& input) {
	return this->operator()(input);
}

template<typename T>
inline Job<T> Nexus<T>::get(const size_t input) {
	return this->operator()(input);
}


// ------------------------------------------------------------------------------------------

template<typename T>
inline size_t Nexus<T>::size() const
{
	return m_inst_task_count;
}

template<typename T>
inline void Nexus<T>::wait_all() const
{
	while (m_task_deque.size()) this->sleep(1);
	while (CPU_Threads::Threads_Used > 0) this->sleep(1);
}

template<typename T>
inline void Nexus<T>::clear()
{
	m_inst_job_xm->clear();
	m_inst_task_count = 0;
}

template<typename T>
inline void Nexus<T>::sleep(unsigned int extent) const
{
#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
	::Sleep(extent);
#else
	::usleep(extent);
#endif
}
