#pragma once

#include "SharedPtr/SharedPtrObj.h"
#include "SharedPtr/SharedPtrArr.h"
#include "SharedPtr/SharedPtrPtr.h"
#include "RawMapping.h"

#ifndef __XPTR__
#define __XPTR__

template<typename T>
using xp = RA::SharedPtr<T>;

template<class T>
using sxp = std::shared_ptr<T>;

#define MKP RA::MakeShared
#define MPVA(_PTR_, ...) _PTR_ = RA::MakeShared<decltype(_PTR_)>(__VA_ARGS__)
#define StdMakePtr std::make_shared
#endif

namespace RA
{
    template <class T> struct IsTypeSharedPtr : std::false_type {};
    template <class T> struct IsTypeSharedPtr<RA::SharedPtr<T>> : std::true_type {};
}

namespace RA
{
    template<typename T> class SharedPtr;

    template <typename T, typename ...A>
    _NODISCARD std::enable_if_t<!IsArray(T) && !IsPointer(T) && !IsFundamental(T), SharedPtr<T>>
        CIN MakeShared(A&&... Args);

#if _HAS_CXX20 // ------------------------------------------------------------------------------------------
    template<typename T> class SharedPtr<T[]>;
    template <typename T>
    _NODISCARD std::enable_if_t<IsArray(T) && !IsPointer(T) &&  IsFundamental(RemoveAllExts(T)), SharedPtr<RemoveExt(T)[]>>
        CIN MakeShared(const xint FnLeng);

    template <typename T, typename ...A>
    _NODISCARD std::enable_if_t<IsArray(T) && !IsPointer(T) && !IsFundamental(RemoveAllExts(T)), SharedPtr<RemoveExt(T)[]>>
        CIN MakeShared(const xint FnLeng, A&&... Args);
#else // ---------------------------------------------------------------------------------------------------
    template<typename T> class SharedPtr<T*>;
    template <typename PA>
    _NODISCARD std::enable_if_t<IsPointer(PA) && IsFundamental(RemovePtr(PA)), SharedPtr<RemovePtr(PA)*>>
        CIN MakeShared(const xint FnLeng);

    template <typename PA, typename ...A>
    _NODISCARD std::enable_if_t<IsPointer(PA) && !IsFundamental(RemovePtr(PA)), SharedPtr<RemovePtr(PA)*>>
        CIN MakeShared(const xint FnLeng, A&&... Args);
#endif
}
// =========================================================================================================
// =========================================================================================================

template <typename T, typename ...A>
_NODISCARD std::enable_if_t<!IsArray(T) && !IsPointer(T) && !IsFundamental(T), RA::SharedPtr<T>>
CIN RA::MakeShared(A&&... Args)
{
    return std::make_shared<T>(std::forward<A>(Args)...);
}

#if _HAS_CXX20 
template <typename T>
_NODISCARD std::enable_if_t<IsArray(T) && !IsPointer(T) && IsFundamental(RemoveAllExts(T)), RA::SharedPtr<RemoveExt(T)[]>>
CIN RA::MakeShared(const xint FnLeng)
{
    RA::SharedPtr<RemoveExt(T)[]> Ptr = std::_Make_shared_unbounded_array<T>(FnLeng + 1);
    Ptr.__SetSize__(FnLeng + 1);
    for (auto& Val : Ptr)
        Val = 0;
    return Ptr;
}

template <typename T, typename ...A>
_NODISCARD std::enable_if_t<IsArray(T) && !IsPointer(T) && !IsFundamental(RemoveAllExts(T)), RA::SharedPtr<RemoveExt(T)[]>>
CIN RA::MakeShared(const xint FnLeng, A&&... Args)
{
    RA::SharedPtr<RemoveExt(T)[]> Ptr = std::_Make_shared_unbounded_array<T>(FnLeng);
    Ptr.__SetSize__(FnLeng);
    if constexpr (sizeof...(Args) > 0)
    {
        Ptr.__SetInitialized__(true);
        for (auto& Elem : Ptr)
            Elem.Construct(std::forward<A>(Args)...);
    }
    return Ptr;
}
#else // ---------------------------------------------------------------------------------------------------
template <typename PA>
_NODISCARD std::enable_if_t<IsPointer(PA) && IsFundamental(RemovePtr(PA)), RA::SharedPtr<RemovePtr(PA)*>>
CIN RA::MakeShared(const xint FnLeng)
{
    return RA::SharedPtr<RemovePtr(PA)*>(FnLeng + 1);
}

template <typename PA, typename ...A>
_NODISCARD std::enable_if_t<IsPointer(PA) && !IsFundamental(RemovePtr(PA)), RA::SharedPtr<RemovePtr(PA)*>>
CIN RA::MakeShared(const xint FnLeng, A&&... Args)
{
    return RA::SharedPtr<RemovePtr(PA)*>(FnLeng, std::forward<A>(Args)...);
}
#endif
// =========================================================================================================
