#pragma once

#include "RawMapping.h"
#include "Memory.h"

template<typename T, typename enabler_t = void> class Task;

#define TaskVoidAPI        Task<T, typename std::enable_if_t< IsSame(T, void)>>
#define TaskValueAPI       Task<T, typename std::enable_if_t<!IsSame(T, void) && !IsSharedPtr(T) && !IsFundamental(T)>>
#define TaskFundamentalAPI Task<T, typename std::enable_if_t<!IsSame(T, void) && !IsSharedPtr(T) &&  IsFundamental(T)>>
#define TaskVirtualAPI     Task<T, typename std::enable_if_t<!IsSame(T, void) &&  IsSharedPtr(T) && !IsFundamental(T)>>

template<typename T> class TaskVoidAPI;
template<typename T> class TaskValueAPI;
template<typename T> class TaskFundamentalAPI;
template<typename T> class TaskVirtualAPI;

// Task<xp<T>> spcialization wouldn't work because then you would need to specify Task<xp<T>> or Task<T> everywhere
// and that wouldn't work when making a queue of Task<T>

using ExceptionPtr = std::exception_ptr;

template<typename T>
class Grind
{
protected:
    VIR RIN xint         GetID()         const = 0;
    VIR RIN void         RunTask() = 0;
    VIR RIN bool         BxRemoved()     const = 0;
    VIR RIN T&           GetValue()      const = 0;
    VIR RIN xp<T>        GetValuePtr()   const = 0;
    VIR RIN bool         IsDone()        const = 0;
    VIR RIN std::string&     GetName()       const = 0;
    VIR RIN xp<std::string>  GetNamePtr()    const = 0;
    VIR RIN void         TestException() const = 0;

    VIR RIN ExceptionPtr Exception() const = 0;

    VIR RIN bool operator> (const Grind<T>& Other) const = 0;
    VIR RIN bool operator< (const Grind<T>& Other) const = 0;
    VIR RIN bool operator==(const Grind<T>& Other) const = 0;
};