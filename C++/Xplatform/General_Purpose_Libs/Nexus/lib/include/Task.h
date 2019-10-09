#pragma once

#include<functional>

#include "xstring.h"

template<typename T>
class Task
{
	xstring* m_name = nullptr;
	std::function<T()> m_method;
	bool m_blank = false;

public:
	Task();
	Task(      Task&& task) noexcept;
	Task(const Task&  task);
	Task(      std::function<T()>&& i_method);
	Task(const std::function<T()>&  i_method);
	Task(      std::function<T()>&& i_method,       xstring&& i_name);
	Task(const std::function<T()>&  i_method, const xstring&  i_name);
	~Task();

	bool blank() const;
	bool has_name() const;
	const xstring* name_ptr() const;
	T operator()();
};


template<typename T>
inline Task<T>::Task()
{
	m_blank = true;
}
// ----------------------------------------------------------------------------------------------------

template<typename T>
inline Task<T>::Task(Task&& task) noexcept
{
	if (task.has_name())
		m_name = new xstring(*task.name_ptr());
	m_method = task.m_method;
}

template<typename T>
inline Task<T>::Task(const Task& task)
{
	if (task.has_name())
		m_name = new xstring(*task.name_ptr());
	m_method = task.m_method;
}
// ----------------------------------------------------------------------------------------------------
template<typename T>
inline Task<T>::Task(std::function<T()>&& i_method) : m_method(i_method)
{   }
template<typename T>
inline Task<T>::Task(const std::function<T()>& i_method) : m_method(i_method)
{   }
// ----------------------------------------------------------------------------------------------------
template<typename T>
inline Task<T>::Task(std::function<T()>&& i_method, xstring&& i_name): m_method(i_method)
{
	m_name = new xstring(i_name);
}
template<typename T>
inline Task<T>::Task(const std::function<T()>& i_method, const xstring& i_name): m_method(i_method)
{
	m_name = new xstring(i_name);
}
// ----------------------------------------------------------------------------------------------------

template<typename T>
inline Task<T>::~Task()
{
	if(m_name != nullptr)
		delete m_name;
}

template<typename T>
inline bool Task<T>::blank() const
{
	return m_blank;
}

template<typename T>
inline bool Task<T>::has_name() const
{
	return (m_name != nullptr) ? true : false;
}

template<typename T>
inline const xstring* Task<T>::name_ptr() const
{
	return m_name;
}

template<typename T>
inline T Task<T>::operator()()
{
	return m_method();
}
