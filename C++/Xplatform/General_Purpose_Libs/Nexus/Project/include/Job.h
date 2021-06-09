#pragma once 
#pragma warning( disable : 26495 ) // for allowing to make a Job class without initializing all vars

#include<iostream>
#include<exception>
#include<functional>
#include<future>
#include<thread>
#include<utility>

#include "NX_Threads.h"
#include "Task.h"

template<typename T>
class Job : protected NX_Threads
{
    Task<T> m_task;
    T m_value;
    
    std::exception_ptr m_exc_ptr = nullptr;
    
    size_t m_index = 0;
    bool m_done = false;
    bool m_removed = false;
    static std::string Default_STR;

public:
    Job(); // required by Linux
    Job(      Task<T>&& task, size_t index);
    Job(const Task<T>&  task, size_t index);
    void Init();
    Task<T> GetTask() const;
    const Task<T>* TaskPtr() const;
    T Move();
    T GetValue() const;
    T* GetValuePtr() const;
    std::exception_ptr Exception() const;
    void ThrowException() const;
    bool IsDone() const;
    size_t GetIndex() const;

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
    m_task(std::move(task)), m_index(std::move(index))
{
}

template<typename T>
inline Job<T>::Job(const Task<T>& task, size_t index) : 
    m_task(task), m_index(index)
{
}

template<typename T>
inline void Job<T>::Init()
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
    s_Threads_Used--;
    m_done = true;
}

template<typename T>
inline Task<T> Job<T>::GetTask() const
{
    return m_task;
}

template<typename T>
inline const Task<T>* Job<T>::TaskPtr() const
{
    if (m_removed)
        throw std::runtime_error("The Task Object Has Been Moved!\n");
    return &m_task;
}

template<typename T>
inline T Job<T>::GetValue() const 
{
    if (m_removed)
        throw std::runtime_error("The Task Object Has Been Moved!\n");
    return m_value;
}


template<typename T>
inline T Job<T>::Move() 
{
    m_removed = true;
    return std::move(m_value);
}

template<typename T>
inline T* Job<T>::GetValuePtr() const
{
    if (m_removed)
        throw std::runtime_error("The Task Object Has Been Moved!\n");
    return &m_value;
}

template<typename T>
inline std::exception_ptr Job<T>::Exception() const
{
    return m_exc_ptr;
}

template<typename T>
inline void Job<T>::ThrowException() const
{
    if(m_exc_ptr != nullptr)
        std::rethrow_exception(m_exc_ptr);
}

template<typename T>
inline bool Job<T>::IsDone() const
{
    return m_done;
}

template<typename T>
inline size_t Job<T>::GetIndex() const
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
    out << job.GetValue();
    return out;
}
