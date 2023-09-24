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
class TaskFundamentalAPI : protected RA::Threads
{
    const xint             IDX;
    xp<std::string>        MsNamePtr = nullptr; // it's constexpr

    T                      MoValue;
    std::function<T()>     MfMethod;

    std::exception_ptr     MoException = nullptr;
    bool                   MbDone = false;
    bool                   MbRemoved = false;

public:
    // Task(); // required by Linux
    RIN Task(const Task<T>& Other) = delete; // Use Shared Pointer
    RIN void operator=(const Task<T>& Other) = delete; // Use Shared Pointer

    RIN Task(const xint FIDX,                                  std::function<T()>&& FfMethod);
    RIN Task(const xint FIDX, const xp<std::string>& FsKeyPtr, std::function<T()>&& FfMethod);


    RIN xint        GetID()               const { return IDX; }
    RIN void        RunTask();

    RIN bool        BxRemoved()           const { return MbRemoved; }
    RIN bool        IsDone()              const { return MbDone; }

    RIN        T&   GetValue()                  { return  MoValue; }
    RIN CST    T&   GetValue()            const { return  MoValue; }
    
    RIN        std::string& GetName()           { return *MsNamePtr; }
    RIN     xp<std::string> GetNamePtr()        { return  MsNamePtr; }
    RIN CST    std::string& GetName()     const { return *MsNamePtr; }
    RIN CST xp<std::string> GetNamePtr()  const { return  MsNamePtr; }

    RIN ExceptionPtr    Exception()       const { return MoException; }
    RIN void            TestException()   const;

    RIN bool operator> (const Task<T>& Other) const;
    RIN bool operator< (const Task<T>& Other) const;
    RIN bool operator==(const Task<T>& Other) const;
};

template<typename T>
RIN TaskFundamentalAPI::Task(xint FIDX, std::function<T()>&& FfMethod) :
    IDX(FIDX), MfMethod(FfMethod)
{
}

template<typename T>
RIN TaskFundamentalAPI::Task(xint FIDX, const xp<std::string>& FsKeyPtr, std::function<T()>&& FfMethod) :
    IDX(FIDX), MsNamePtr(FsKeyPtr), MfMethod(FfMethod)
{
}

template<typename T>
RIN void TaskFundamentalAPI::RunTask()
{
    if (MbDone && MbRemoved == false)
        return;

    RA::Threads::TotalTasksCounted++;
    try
    {
        MoValue = MfMethod();
    }
    catch (...) {
        MoException = std::current_exception();
    }

    MbDone = true;
}

template<typename T>
RIN void TaskFundamentalAPI::TestException() const
{
    if (MoException != nullptr)
        std::rethrow_exception(MoException);
}

template<typename T>
RIN bool TaskFundamentalAPI::operator>(const Task<T>& Other) const
{
    return IDX > Other.IDX;
}

template<typename T>
RIN bool TaskFundamentalAPI::operator<(const Task<T>& Other) const
{
    return IDX < Other.IDX;
}

template<typename T>
RIN bool TaskFundamentalAPI::operator==(const Task<T>& Other) const
{
    return IDX == Other.IDX;
}
