﻿#pragma once 
#pragma warning( disable : 26495 ) // for allowing to make a Job class without initializing all vars

#include<iostream>
#include<exception>
#include<functional>
#include<future>
#include<thread>
#include<utility>

#include "SharedPtr.h"
#include "Threads.h"
#include "Task.h"

template<typename T>
class Job : protected RA::Threads
{
    RA::SharedPtr<Task<T>> MoTaskPtr  = nullptr;
    T*                     MoValuePtr = nullptr;
    
    std::exception_ptr  m_exc_ptr = nullptr;
    
    xint              Idx = 0;
    bool                MbDone = false;
    bool                MbRemoved = false;
    static std::string  Default_STR;

public:
    // Job(); // required by Linux
    Job(const Job<T>& Other) = delete; // Use Shared Pointer
    Job(const RA::SharedPtr<Task<T>>&  task, xint index);
    ~Job();
    void Run();
    const Task<T>& GetTask() const;
    const RA::SharedPtr<Task<T>> TaskPtr() const;
    T    Move();
    bool BxRemoved() const { return MbRemoved; }
    T&   GetValue();
    T*   GetValuePtr();
    std::exception_ptr Exception() const;
    void TestException() const;
    bool IsDone() const;
    xint GetIndex() const;

    void operator=(const Job<T>& Other) const = delete;

    bool operator> (const Job<T>& Other) const;
    bool operator< (const Job<T>& Other) const;
    bool operator==(const Job<T>& Other) const;
};
template<typename T> std::string Job<T>::Default_STR = "";

template<typename T>
inline Job<T>::Job(const RA::SharedPtr<Task<T>>& task, xint index) : 
    MoTaskPtr(task), Idx(index)
{
}

template<typename T>
inline Job<T>::~Job()
{
    DeleteObj(MoValuePtr);
}

template<typename T>
inline void Job<T>::Run()
{
    if (MbDone && MbRemoved == false)
        return;

    RA::Threads::Used++;
    RA::Threads::TotalTasksCounted++;
    try {
        DeleteObj(MoValuePtr);
        MoValuePtr = new T(MoTaskPtr.Get().RunTask());
    }
    catch (const char*) {
        m_exc_ptr = std::current_exception();
    }
    catch (const std::exception&) {
        m_exc_ptr = std::current_exception();
    }

    RA::Threads::Used--;
    MbDone = true;
}

template<typename T>
inline const Task<T>& Job<T>::GetTask() const
{
    return MoTaskPtr.Get();
}

template<typename T>
inline const RA::SharedPtr<Task<T>> Job<T>::TaskPtr() const
{
    if (MbRemoved)
        throw std::runtime_error("The Task Object Has Been Moved!\n");
    return MoTaskPtr;
}

template<typename T>
inline T Job<T>::Move()
{
    if (MbRemoved || !MoValuePtr)
        throw "NullPtr!";
    MbRemoved = true;
    return std::move(*MoValuePtr);
}

template<typename T>
inline T& Job<T>::GetValue() 
{
    if (MbRemoved)
        throw std::runtime_error("The Task Object Has Been Moved!\n");
    if (!MoValuePtr)
        throw "NullPtr!";
    return *MoValuePtr;
}

template<typename T>
inline T* Job<T>::GetValuePtr()
{
    if (MbRemoved)
        throw std::runtime_error("The Task Object Has Been Moved!\n");
    return MoValuePtr;
}

template<typename T>
inline std::exception_ptr Job<T>::Exception() const
{
    return m_exc_ptr;
}

template<typename T>
inline void Job<T>::TestException() const
{
    if(m_exc_ptr != nullptr)
        std::rethrow_exception(m_exc_ptr);
}

template<typename T>
inline bool Job<T>::IsDone() const
{
    return MbDone;
}

template<typename T>
inline xint Job<T>::GetIndex() const
{
    return Idx;
}

template<typename T>
inline bool Job<T>::operator>(const Job<T>& Other) const
{
    return Idx > Other.Idx;
}

template<typename T>
inline bool Job<T>::operator<(const Job<T>& Other) const
{
    return Idx < Other.Idx;
}

template<typename T>
inline bool Job<T>::operator==(const Job<T>& Other) const
{
    return Idx == Other.Idx;
}

// Below are NOT Member Functions
template<typename T>
std::ostream& operator<<(std::ostream& out, Job<T>& job)
{
    out << job.GetValue();
    return out;
}
