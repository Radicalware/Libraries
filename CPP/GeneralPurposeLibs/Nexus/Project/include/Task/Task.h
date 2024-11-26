#pragma once

#include "RawMapping.h"
#include "Memory.h"

template<typename T, typename enabler_t = void> class Task;

#define TaskVoidAPI        Task<T, typename std::enable_if_t< BxSameType(T, void)>>
#define TaskValueAPI       Task<T, typename std::enable_if_t<!BxSameType(T, void) && !BxSharedPtr(T) && !BxFundamental(T)>>
#define TaskFundamentalAPI Task<T, typename std::enable_if_t<!BxSameType(T, void) && !BxSharedPtr(T) &&  BxFundamental(T)>>
#define TaskVirtualAPI     Task<T, typename std::enable_if_t<!BxSameType(T, void) &&  BxSharedPtr(T) && !BxFundamental(T)>>

template<typename T> class TaskVoidAPI;
template<typename T> class TaskValueAPI;
template<typename T> class TaskFundamentalAPI;
template<typename T> class TaskVirtualAPI;

using ExceptionPtr = std::exception_ptr;

