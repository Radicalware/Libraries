#pragma once

#include<iostream>
#include<initializer_list>
#include<utility>

#include<future>
#include<thread>
#include<mutex>
#include<condition_variable>
//#include<xtr1common>

#include<deque>
#include<functional>
#include<type_traits>

#include "xvector.h"
#include "xstring.h"
#include "xmap.h"

#include "Task.h"
#include "Job.h"

// =========================================================================================
namespace util {
	template <typename T>
	using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;
}

template<typename T>
class Nexus
{
private:
	static int     Thread_Count;
	static size_t  Inst_Count;
	static size_t  Task_Count;
	static bool    Finish_Tasks;

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
	Job<T> get(const xstring& val);
	Job<T> get(const size_t val);

	const size_t task_count()   const;
	const size_t inst_count()   const;
	const size_t thread_count() const;
}; 
template<typename T> size_t  Nexus<T>::Task_Count = 0;
template<typename T> size_t  Nexus<T>::Inst_Count = 0;
template<typename T> bool    Nexus<T>::Finish_Tasks = false;
template<typename T> int     Nexus<T>::Thread_Count = std::thread::hardware_concurrency();

// =========================================================================================

template<typename T>
inline void Nexus<T>::TaskLooper(int thread_idx)
{
	while (true) {
		std::unique_lock<std::mutex> lock(m_mutex);
		m_sig_deque.wait(lock, [this]() {
			return Nexus<T>::Finish_Tasks || m_task_deque.size();
		});

		if (m_task_deque.empty())
			return;
		
		(*m_inst_job_xm).add_pair(m_inst_task_count, Job<T>(std::move(m_task_deque.front()), Nexus<T>::Task_Count));

		const Task<T>* latest_task = (*m_inst_job_xm)[m_inst_task_count].task_ptr();
		if (latest_task->has_name()) 
			m_str_inst_xm.add_pair(latest_task->name_ptr(), m_inst_task_count);

		Nexus<T>::Task_Count++;
		m_inst_task_count++;
		m_task_deque.pop_front();
		m_sig_get.notify_all();
	}
}

template<typename T>
Nexus<T>::Nexus() 
{
	m_inst_job_xm = new xmap<size_t, Job<T>>;
	Nexus<T>::Inst_Count++;
	m_threads.reserve(Nexus<T>::Thread_Count);
	for (int i = 0; i < Nexus<T>::Thread_Count; ++i)
		m_threads.emplace_back(std::bind(&Nexus<T>::TaskLooper, this, i));
}

template<typename T>
Nexus<T>::~Nexus()
{
	{
		std::unique_lock <std::mutex> lock(m_mutex);
		Nexus<T>::Finish_Tasks = true;
		m_sig_deque.notify_all();
	}
	m_threads.proc([](auto& t) { t.join(); });
}

template<typename T>
template<typename F, typename ...A>
inline void Nexus<T>::add_job(const xstring& key, F&& function, A&&... Args)
{
	std::lock_guard<std::mutex> glock(m_get_mutex);
	auto binded_function = std::bind(std::forward<F>(function), std::forward<A>(Args)...);
	std::lock_guard <std::mutex> lock(m_mutex);
	m_task_deque.emplace_back(std::move(binded_function), key);
	m_sig_deque.notify_one();
	m_sig_get.notify_all();
}

template<typename T>
template<typename F, typename ...A>
inline void Nexus<T>::add_job(const char* key, F&& function, A&& ...Args)
{
	std::lock_guard<std::mutex> glock(m_get_mutex);
	auto binded_function = std::bind(std::forward<F>(function), std::forward<A>(Args)...);
	std::lock_guard <std::mutex> lock(m_mutex);
	m_task_deque.emplace_back(std::move(binded_function), xstring(key));
	m_sig_deque.notify_one();
	m_sig_get.notify_all();
}

template<typename T>
template <typename F, typename... A>
inline auto Nexus<T>::add_job(F&& function, A&& ... Args)->
	std::enable_if_t<!std::is_same_v<util::remove_cvref_t<F>, xstring>, void>
{
	std::lock_guard<std::mutex> glock(m_get_mutex);
	auto binded_function = std::bind(std::forward<F>(function), std::forward<A>(Args)...);
	std::lock_guard <std::mutex> lock(m_mutex);
	m_task_deque.emplace_back(std::move(binded_function));
	m_sig_deque.notify_one();
	m_sig_get.notify_all();
}

template<typename T>
inline Job<T> Nexus<T>::get(const xstring& input) 
{
	std::unique_lock<std::mutex> glock(m_get_mutex);
	if(m_task_deque.size())
		m_sig_get.wait(glock);

	if (!m_str_inst_xm.has(input)) {
		for (const Task<T>& t : m_task_deque) {
			if (*t.name_ptr() == input)
				break;
		}
		throw std::runtime_error("Nexus Key Not Found!");
	}
	while (!m_str_inst_xm.has(input)) continue;
	size_t loc = m_str_inst_xm[input];
	while (!(*m_inst_job_xm)[loc].done()) continue;

	(*m_inst_job_xm)[loc].rethrow_exception();
	return (*m_inst_job_xm)[loc];
}

template<typename T>
inline Job<T> Nexus<T>::get(const size_t input) 
{
	std::unique_lock<std::mutex> glock(m_get_mutex);
	if (m_task_deque.size())
		m_sig_get.wait(glock);

	if (m_inst_task_count + m_task_deque.size() < input + 1)
		throw std::runtime_error("Requested Job is Out of Range\n");

	while (!(*m_inst_job_xm)(input)) continue;
	while (!(*m_inst_job_xm)[input].done()) continue;

	(*m_inst_job_xm)[input].rethrow_exception();
	return (*m_inst_job_xm)[input];
}

template<typename T>
inline const size_t Nexus<T>::task_count() const
{
	return Nexus<T>::Task_Count;
}

template<typename T>
inline const size_t Nexus<T>::inst_count() const
{
	return Nexus<T>::Inst_Count;
}

template<typename T>
inline const size_t Nexus<T>::thread_count() const
{
	return Nexus<T>::Thread_Count;
}

