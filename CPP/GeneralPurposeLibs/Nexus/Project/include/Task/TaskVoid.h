#pragma once

#pragma warning( disable : 26495 ) // for allowing to make a Task class without initializing all vars

#include "Task/Task.h"

#include<iostream>
#include<exception>
#include<functional>
#include<future>
#include<thread>
#include<utility>
#include<type_traits>


#include "RawMapping.h"
#include "SharedPtr.h"
#include "Threads.h"
#include "Task.h"

template<typename T>
class TaskVoidAPI : protected RA::Threads
{
    const xint             IDX;
    std::function<void()>  MfMethod;

    std::exception_ptr     MoException = nullptr;
    bool                   MbDone = false;
    bool                   MbRemoved = false;

public:
    // Task(); // required by Linux
    RIN Task(const Task<void>& Other) = delete; // Use Shared Pointer
    RIN void operator=(const Task<void>& Other) = delete; // Use Shared Pointer

    RIN Task(const xint FIDX, std::function<void()>&& FfMethod);


    VIR RIN xint        GetID()               const { return IDX; }
    VIR RIN void        RunTask();

    VIR RIN bool        BxRemoved()           const { return MbRemoved; }
    VIR RIN bool        IsDone()              const { return MbDone; }
    
    VIR RIN ExceptionPtr Exception()          const { return MoException; }
    VIR RIN void         TestException()      const;

    RIN bool operator> (const Task<T>& Other) const;
    RIN bool operator< (const Task<T>& Other) const;
    RIN bool operator==(const Task<T>& Other) const;
};

template<typename T>
RIN TaskVoidAPI::Task(const xint FIDX, std::function<void()>&& FfMethod) :
    IDX(FIDX), MfMethod(FfMethod)
{
}

template<typename T>
RIN void TaskVoidAPI::RunTask()
{
    if (MbDone && MbRemoved == false)
        return;

    RA::Threads::TotalTasksCounted++;
    try
    {
        MfMethod();
    }
    catch (...) {
        MoException = std::current_exception();
    }

    MbDone = true;
}

template<typename T>
RIN void TaskVoidAPI::TestException() const
{
    if (MoException != nullptr)
        std::rethrow_exception(MoException);
}

template<typename T>
RIN bool TaskVoidAPI::operator>(const Task<T>& Other) const
{
    return IDX > Other.IDX;
}

template<typename T>
RIN bool TaskVoidAPI::operator<(const Task<T>& Other) const
{
    return IDX < Other.IDX;
}

template<typename T>
RIN bool TaskVoidAPI::operator==(const Task<T>& Other) const
{
    return IDX == Other.IDX;
}
