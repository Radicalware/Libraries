#pragma once 

#include<iostream>
#include<exception>
#include<functional>
#include<future>
#include<thread>

#include "CPU_Threads.h"
#include "Task.h"

template<typename T>
class Job : protected CPU_Threads
{
	Task<T> m_task;
	T m_value;
	
	std::exception_ptr m_exc_ptr = nullptr;
	
	size_t m_index;
	bool m_done = false;
	static std::string Default_STR;
public:
	Job();
	Job(      Task<T>&& task, size_t index);
	Job(const Task<T>&  task, size_t index);
	void init();
	Task<T> task() const;
	const Task<T>* task_ptr() const;
	T value() const;
	std::exception_ptr exception() const;
	void rethrow_exception() const;
	bool done() const;
	size_t index() const;

	bool operator> (const Job<T> other) const;
	bool operator< (const Job<T> other) const;
	bool operator==(const Job<T> other) const;
};
template<typename T> std::string Job<T>::Default_STR = "";

template<typename T>
inline Job<T>::Job()
{
}

template<typename T>
inline Job<T>::Job(Task<T>&& task, size_t index) : 
	m_task(task), m_index(index)
{
}

template<typename T>
inline Job<T>::Job(const Task<T>& task, size_t index) : 
	m_task(task), m_index(index)
{
}

template<typename T>
inline void Job<T>::init()
{
	try {
		m_value = m_task();
	}
	catch (const char*) {
		m_exc_ptr = std::current_exception();
	}
	catch (const std::exception&) {
		m_exc_ptr = std::current_exception();
	}
	CPU_Threads::Threads_Used--;
	m_done = true;
}

template<typename T>
inline Task<T> Job<T>::task() const
{
	return m_task;
}

template<typename T>
inline const Task<T>* Job<T>::task_ptr() const
{
	return &m_task;
}

template<typename T>
inline T Job<T>::value() const
{
	return m_value;
}

template<typename T>
inline std::exception_ptr Job<T>::exception() const
{
	return m_exc_ptr;
}

template<typename T>
inline void Job<T>::rethrow_exception() const
{
	if(m_exc_ptr != nullptr)
		std::rethrow_exception(m_exc_ptr);
}

template<typename T>
inline bool Job<T>::done() const
{
	return m_done;
}

template<typename T>
inline size_t Job<T>::index() const
{
	return m_index;
}

template<typename T>
inline bool Job<T>::operator>(const Job<T> other) const
{
	return m_index > other.m_index;
}

template<typename T>
inline bool Job<T>::operator<(const Job<T> other) const
{
	return m_index < other.m_index;
}

template<typename T>
inline bool Job<T>::operator==(const Job<T> other) const
{
	return m_index == other.m_index;
}

// Below are NOT Member Functions
template<typename T>
std::ostream& operator<<(std::ostream& out, const Job<T>& job)
{
	out << job.value();
	return out;
}
