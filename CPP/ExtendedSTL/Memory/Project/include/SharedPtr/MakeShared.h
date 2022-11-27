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
    inline RA::MakeShared(const uint FnLeng)
{
    return RA::SharedPtr<PA>(FnLeng);
}

template <typename PA, typename ...A>
_NODISCARD std::enable_if_t<IsPointer(PA) && IsClass(RemovePtr(PA)), RA::SharedPtr<PA>>
    inline RA::MakeShared(const uint FnLeng, A&&... Args) 
{
    return RA::SharedPtr<PA>(FnLeng, Args...);
}

template<typename T>
inline std::enable_if_t<IsClass(RemovePtr(T)), RA::SharedPtr<T*>>
RA::MakeSharedBuffer(const uint FnLeng, const uint FnUnitByteSize)
{
    const auto BufferSize = (FnLeng * FnUnitByteSize + sizeof(uint));
    T* LvArray = (T*)malloc(BufferSize);
    auto Ptr = RA::SharedPtr<T*>(LvArray);
    Ptr.__SetSize__(FnLeng);
    return Ptr;
}

template<typename T>
inline std::enable_if_t<IsFundamental(RemovePtr(T)), RA::SharedPtr<T*>>
    RA::MakeSharedBuffer(const uint FnLeng, const uint FnUnitByteSize)
{
    const auto BufferSize = (FnLeng * FnUnitByteSize + sizeof(uint));
    T* LvArray = (T*)malloc(BufferSize);
    for (auto* Ptr = LvArray; Ptr < LvArray + FnLeng; Ptr++)
        *Ptr = '\0';
    auto Ptr = RA::SharedPtr<T*>(LvArray);
    Ptr.__SetSize__(FnLeng);
    return Ptr;
}

template<typename T, typename D>
inline RA::SharedPtr<T*>
     RA::MakeSharedBuffer(const uint FnLeng, const uint FnUnitByteSize, D&& FfDestructor)
{
    const auto BufferSize = (FnLeng * (FnUnitByteSize + sizeof(uint)));
    T* LvArray = (T*)malloc(BufferSize);
    auto Ptr = RA::SharedPtr<T*>(LvArray);
    Ptr.SetDestructor(FfDestructor);
    Ptr.__SetSize__(FnLeng);
    return Ptr;
}

#ifndef __XPTR__
#define __XPTR__

template<typename T>
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