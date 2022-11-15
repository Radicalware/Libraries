#pragma once

#include "SharedPtr/BaseSharedPtr.h"

template <typename T, typename ...A>
_NODISCARD std::enable_if_t<!IsArray(T) && !IsPointer(T), RA::SharedPtr<T>>
    inline RA::MakeShared(A&&... Args) 
{
    return std::make_shared<T>(std::forward<A>(Args)...);
}

template <typename PA>
_NODISCARD std::enable_if_t<IsPointer(PA) && !IsClass(RemovePtr(PA)), RA::SharedPtr<PA>>
    inline RA::MakeShared(const size_t FnSize)
{
    return RA::SharedPtr<RemovePtr(PA)*>(FnSize);
}

template <typename PA, typename ...A>
_NODISCARD std::enable_if_t<IsPointer(PA) && IsClass(RemovePtr(PA)), RA::SharedPtr<PA>>
    inline RA::MakeShared(const size_t FnSize, A&&... Args) 
{
    return RA::SharedPtr<RemovePtr(PA)*>(FnSize, Args...);
}

#ifndef __XPTR__
#define __XPTR__

template<class T>
using xp = RA::SharedPtr<T>;

template<class T>
using sxp = std::shared_ptr<T>;

#define MKP RA::MakeShared
#define StdMakePtr std::make_shared
#endif

namespace RA
{
    template <class T> struct IsTypeSharedPtr : std::false_type {};
    template <class T> struct IsTypeSharedPtr<RA::SharedPtr<T>> : std::true_type {};
}