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
class TaskValueAPI : protected RA::Threads
{
    const xint             IDX;
    xp<std::string>        MsNamePtr = nullptr; // it's constexpr

    xp<T>                  MoValuePtr = nullptr;
    std::function<T()>     MfMethodValue; // breaks on abstract class
    std::function<xp<T>()> MfMethodPtr;

    std::exception_ptr     MoException = nullptr;
    bool                   MbDone = false;
    bool                   MbRemoved = false;

public:
    // Task(); // required by Linux
    RIN Task(const Task<T>& Other) = delete; // Use Shared Pointer
    RIN void operator=(const Task<T>& Other) = delete; // Use Shared Pointer

    template<class TT = T, typename std::enable_if<!IsSharedPtr(TT), bool>::type = 0>
    RIN Task(const xint FIDX,                                  std::function<T()>&& FfMethod);
    template<class TT = T, typename std::enable_if<!IsSharedPtr(TT), bool>::type = 0>
    RIN Task(const xint FIDX, const xp<std::string>& FsKeyPtr, std::function<T()>&& FfMethod);

    RIN Task(const xint FIDX,                                  std::function<xp<T>()>&& FfMethod);
    RIN Task(const xint FIDX, const xp<std::string>& FsKeyPtr, std::function<xp<T>()>&& FfMethod);

    RIN xint        GetID()               const { return IDX; }
    RIN void        RunTask();

    RIN bool        BxRemoved()           const { return MbRemoved; }
    RIN bool        IsDone()              const { return MbDone; }

    RIN        T&   GetValue()                  { return *MoValuePtr; }
    RIN     xp<T>   GetValuePtr()               { return  MoValuePtr; }
    RIN CST    T&   GetValue()            const { return *MoValuePtr; }
    RIN CST xp<T>   GetValuePtr()         const { return  MoValuePtr; }
    
    RIN        std::string& GetName()           { return *MsNamePtr; }
    RIN     xp<std::string> GetNamePtr()        { return  MsNamePtr; }
    RIN CST    std::string& GetName()     const { return *MsNamePtr; }
    RIN CST xp<std::string> GetNamePtr()  const { return  MsNamePtr; }

    RIN ExceptionPtr Exception()          const { return MoException; }
    RIN void         TestException()      const;

    RIN bool operator> (const Task<T>& Other) const;
    RIN bool operator< (const Task<T>& Other) const;
    RIN bool operator==(const Task<T>& Other) const;
};

template<typename T>
template<class TT, typename std::enable_if< !IsSharedPtr(TT), bool>::type>
RIN TaskValueAPI::Task(const xint FIDX, std::function<T()>&& FfMethod) :
    IDX(FIDX), MfMethodValue(FfMethod)
{
}

template<typename T>
template<class TT, typename std::enable_if< !IsSharedPtr(TT), bool>::type>
RIN TaskValueAPI::Task(const xint FIDX, const xp<std::string>& FsKeyPtr, std::function<T()>&& FfMethod) :
    IDX(FIDX), MsNamePtr(FsKeyPtr), MfMethodValue(FfMethod)
{
}

template<typename T>
RIN TaskValueAPI::Task(const xint FIDX, std::function<xp<T>()>&& FfMethod) :
    IDX(FIDX), MfMethodPtr(FfMethod)
{
}


template<typename T>
RIN TaskValueAPI::Task(const xint FIDX, const xp<std::string>& FsKeyPtr, std::function<xp<T>()>&& FfMethod) :
    IDX(FIDX), MsNamePtr(FsKeyPtr), MfMethodPtr(FfMethod)
{
}

template<typename T>
RIN void TaskValueAPI::RunTask()
{
    if (MbDone && MbRemoved == false)
        return;

    RA::Threads::TotalTasksCounted++;
    try
    {
        if (MfMethodPtr != nullptr)
            MoValuePtr = MfMethodPtr();
        else if (MfMethodValue != nullptr)
              MoValuePtr = RA::MakeShared<T>(MfMethodValue());
        else
            throw "TaskValueAPI::RunTask No Available Method!!";
    }
    catch (...) {
        MoException = std::current_exception();
    }

    MbDone = true;
}

template<typename T>
RIN void TaskValueAPI::TestException() const
{
    if (MoException != nullptr)
        std::rethrow_exception(MoException);
}

template<typename T>
RIN bool TaskValueAPI::operator>(const Task<T>& Other) const
{
    return IDX > Other.IDX;
}

template<typename T>
RIN bool TaskValueAPI::operator<(const Task<T>& Other) const
{
    return IDX < Other.IDX;
}

template<typename T>
RIN bool TaskValueAPI::operator==(const Task<T>& Other) const
{
    return IDX == Other.IDX;
}

// Below are NOT Member Functions
template<typename T>
std::ostream& operator<<(std::ostream& out, const Task<T>& FoTask)
{
    out << FoTask.GetValue();
    return out;
}
