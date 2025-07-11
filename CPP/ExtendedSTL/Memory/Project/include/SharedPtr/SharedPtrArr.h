﻿#pragma once

#include "SharedPtr/BaseSharedPtr.h"

#if _HAS_CXX20

#include "Iterator.h"

#include <functional>
#include <algorithm>
#include <utility>
#include <execution>

namespace RA
{
    template<typename T>
    class SharedPtr<T[]> : public BaseSharedPtr<T[]>, public RA::Iterator<T>
    {
    protected:
        xint MnLeng = 0;
        bool MbInitialized = false;

    public:
        using BaseSharedPtr<T[]>::BaseSharedPtr;
        using BaseSharedPtr<T[]>::operator=;

        RIN ~SharedPtr();

        template<typename F, typename ...A>
        CIN SharedPtr<T[]> Initialize(F&& FfInitialize, A&&... Args);

        template<class TT = T, typename std::enable_if<!BxClass(TT), bool>::type = 0>
        CIN SharedPtr(const xint FnLeng);

        template<typename ...A, class TT = T, typename std::enable_if< BxClass(TT), bool>::type = 0>
        CIN SharedPtr(const xint FnLeng, A&&... Args);

        CIN SharedPtr(SharedPtr<T[]>&& Other);
        CIN SharedPtr(const SharedPtr<T[]>& Other);

        CIN void operator=(SharedPtr<T[]>&& Other);
        CIN void operator=(const SharedPtr<T[]>& Other);

        CIN void Clone(SharedPtr<T[]>&& Other);
        CIN void Clone(const SharedPtr<T[]>& Other);

        _NODISCARD CIN       T* Ptr()       noexcept { return The.get(); }
        _NODISCARD CIN const T* Ptr() const noexcept { return The.get(); }

        _NODISCARD CIN       T* Raw()       noexcept { return The.get(); }
        _NODISCARD CIN const T* Raw() const noexcept { return The.get(); }

        CIN        T& operator[](const xint Idx);
        CIN  const T& operator[](const xint Idx) const;

        CIN auto Size()   const { return MnLeng; }
        CIN auto GetLength() const { return MnLeng; }
        CIN auto GetUnitSize() const { return sizeof(T); }
        CIN auto GetMallocSize() const { return sizeof(T) * MnLeng + sizeof(xint); }

        template<typename F, typename ...A>
        SharedPtr<T[]> RIN Proc(F&& Func, A&&... Args);
        template<typename F, typename ...A>
        SharedPtr<T[]> RIN ProcThreadsSeq(F&& Func);
        template<typename F, typename ...A>
        SharedPtr<T[]> RIN ProcThreadsUnseq(F&& Func);

        template<typename R = T, typename F, typename ...A>
        SharedPtr<R[]> RIN ForEach(F&& Func, A&&... Args);
        template<typename R = T, typename F, typename ...A>
        SharedPtr<R[]> RIN ForEach(F&& Func, A&&... Args) const;

        template<typename R = T, typename F, typename ...A>
        RIN R ForEachAdd(F&& Func, A&&... Args);
        template<typename R = T, typename F, typename ...A>
        RIN R ForEachAdd(F&& Func, A&&... Args) const;

        template<typename R = T, typename F>
        SharedPtr<R[]> RIN ForEachThreadSeq(F&& Func);
        template<typename R = T, typename F>
        SharedPtr<R[]> RIN ForEachThreadSeq(F&& Func) const;

        template<typename R = T, typename F>
        SharedPtr<R[]> RIN ForEachThreadUnseq(F&& Func);
        template<typename R = T, typename F>
        SharedPtr<R[]> RIN ForEachThreadUnseq(F&& Func) const;


        CIN void __SetSize__(const xint FnLeng); // todo: get friend functions working
        CIN void __SetInitialized__(const bool FbInit);
    };
}

template<typename T>
RIN RA::SharedPtr<T[]>::~SharedPtr()
{
    if (!The.MbDestructorSet)
        return;
    The.MbDestructorSet = false;
    if (The.get() != nullptr && MnLeng > 0 && The.use_count() < 1)
    {
        if constexpr (!BxFundamental(RemoveAllExts(T)))
        {
            for (auto& Obj : The)
                Obj.~T();
        }
    }
}

template<typename T>
template<typename F, typename ...A>
CIN RA::SharedPtr<T[]> RA::SharedPtr<T[]>::Initialize(F&& FfInitialize, A&&... Args)
{
    The.MbDestructorSet = true;
    if (MbInitialized)
        throw "RA::SharedPtr<T[]>::Initialize >> Already Initialized!";
    MbInitialized = true;
    for (auto& Obj : The)
        FfInitialize(Obj, std::forward<A>(Args)...);
    return The;
}

template<typename T>
template<class TT, typename std::enable_if<!BxClass(TT), bool>::type>
CIN RA::SharedPtr<T[]>::SharedPtr(const xint FnLeng) :
    RA::BaseSharedPtr<T[]>(new T[FnLeng+1])
{
    The.MbDestructorSet = true;
    The.__SetSize__(FnLeng);
    for (auto* ptr = The.get(); ptr < The.get() + FnLeng + 1; ptr++)
        *ptr = 0;
}

template<typename T>
template<typename ...A, class TT, typename std::enable_if< BxClass(TT), bool>::type>
CIN RA::SharedPtr<T[]>::SharedPtr(const xint FnLeng, A&&... Args) :
    RA::BaseSharedPtr<T[]>(new T[FnLeng])
{
    The.MbDestructorSet = true;
    The.__SetSize__(FnLeng);
    if constexpr (sizeof...(Args) > 0)
    {
        The.__SetInitialized__(true);
        for (auto& Elem : The)
            Elem.Construct(std::forward<A>(Args)...);
    }
}


template<typename T>
CIN RA::SharedPtr<T[]>::SharedPtr(RA::SharedPtr<T[]>&& Other)
{
    The.MbDestructorSet = true;
    Other.MbDestructorSet = false;
    The._Move_construct_from(std::move(Other));
    The.__SetSize__(Other.Size());
    MbInitialized = Other.MbInitialized;
    The.MbDestructorSet = true;
}

template<typename T>
CIN RA::SharedPtr<T[]>::SharedPtr(const RA::SharedPtr<T[]>& Other)
{
    The._Copy_construct_from(Other);
    The.__SetSize__(Other.Size());
    MbInitialized = Other.MbInitialized;
}

template<typename T>
CIN void RA::SharedPtr<T[]>::operator=(RA::SharedPtr<T[]>&& Other)
{
    The.MbDestructorSet = true;
    Other.MbDestructorSet = false;
    The._Move_construct_from(std::move(Other));
    The.__SetSize__(Other.Size());
    MbInitialized = Other.MbInitialized;
    The.MbDestructorSet = true;
}

template<typename T>
CIN void RA::SharedPtr<T[]>::operator=(const RA::SharedPtr<T[]>& Other)
{
    The.MbDestructorSet = true;
    The._Copy_construct_from(Other);
    The.__SetSize__(Other.Size());
    MbInitialized = Other.MbInitialized;
}

template<typename T>
CIN void RA::SharedPtr<T[]>::Clone(RA::SharedPtr<T[]>&& Other)
{
    The = std::move(Other);
}

template<typename T>
CIN void RA::SharedPtr<T[]>::Clone(const RA::SharedPtr<T[]>& Other)
{
    if (!Other)
    {
        The = nullptr;
        MnLeng = 0;
        return;
    }

    if (The.Size() != Other.Size())
    {
        The.~SharedPtr();
        The = RA::SharedPtr<T[]>(Other.Size());
    }

    The.MbDestructorSet = true;
    for (int i = 0; i < The.MnLeng; i++)
        The[i] = Other[i];
}

template<typename T>
CIN T& RA::SharedPtr<T[]>::operator[](const xint Idx)
{
    if (Idx >= MnLeng)
        throw "Idx is out of range!";
    return The.get()[Idx];
}

template<typename T>
CIN const T& RA::SharedPtr<T[]>::operator[](const xint Idx) const
{
    if (Idx >= MnLeng)
        throw "Idx is out of range!";
    return The.get()[Idx];
}

template<typename T>
template<typename F, typename ...A>
RIN RA::SharedPtr<T[]> RA::SharedPtr<T[]>::Proc(F&& Func, A&& ...Args)
{
    for (auto& Elem : The)
        Func(Elem, std::forward<A>(Args)...);
    return The;
}
template<typename T>
template<typename F, typename ...A>
RIN RA::SharedPtr<T[]> RA::SharedPtr<T[]>::ProcThreadsSeq(F&& Func)
{
    std::for_each(
        std::execution::par,
        The.begin(),
        The.end(),
        Func);
    return The;
}
template<typename T>
template<typename F, typename ...A>
RIN RA::SharedPtr<T[]> RA::SharedPtr<T[]>::ProcThreadsUnseq(F&& Func)
{
    std::for_each(
        std::execution::unseq,
        The.begin(),
        The.end(),
        Func);
    return The;
}

template<typename T>
template<typename R, typename F, typename ...A>
RIN RA::SharedPtr<R[]> RA::SharedPtr<T[]>::ForEach(F&& Func, A&& ...Args)
{
    auto Ret = The.GetNew(The.Size());
    for (xint i = 0; i < MnLeng; i++)
        Ret[i] = Func(The[i], std::forward<A>(Args)...);
    return Ret;
}

template<typename T>
template<typename R, typename F, typename ...A>
RIN RA::SharedPtr<R[]> RA::SharedPtr<T[]>::ForEach(F&& Func, A && ...Args) const
{
    auto Ret = The.GetNew(The.Size());
    for (xint i = 0; i < MnLeng; i++)
        Ret[i] = Func(The[i], std::forward<A>(Args)...);
    return Ret;
}

template<typename T>
template<typename R, typename F, typename ...A>
RIN R RA::SharedPtr<T[]>::ForEachAdd(F&& Func, A && ...Args)
{
    R Ret = 0;
    for (xint i = 0; i < MnLeng; i++)
        Ret += Func(The[i], std::forward<A>(Args)...);
    return Ret;
}

template<typename T>
template<typename R, typename F, typename ...A>
RIN R RA::SharedPtr<T[]>::ForEachAdd(F&& Func, A && ...Args)const
{
    R Ret = 0;
    for (xint i = 0; i < MnLeng; i++)
        Ret += Func(The[i], std::forward<A>(Args)...);
    return Ret;
}

// ------------------------------------------------------------------------


template<typename T>
template<typename R, typename F>
RIN RA::SharedPtr<R[]> RA::SharedPtr<T[]>::ForEachThreadSeq(F&& Func)
{
    std::for_each(
        std::execution::par,
        The.begin(),
        The.end(),
        Func
    );
    return The;
}

template<typename T>
template<typename R, typename F>
RIN RA::SharedPtr<R[]> RA::SharedPtr<T[]>::ForEachThreadSeq(F&& Func) const
{
    auto Ret = The.GetNew(The.Size());
    std::for_each(
        std::execution::par,
        Ret.begin(),
        Ret.end(),
        Func
    );
    return Ret;
}

template<typename T>
template<typename R, typename F>
RIN RA::SharedPtr<R[]> RA::SharedPtr<T[]>::ForEachThreadUnseq(F&& Func)
{
    std::for_each(
        std::execution::unseq,
        The.begin(),
        The.end(),
        Func
    );
    return The;
}

template<typename T>
template<typename R, typename F>
RIN RA::SharedPtr<R[]> RA::SharedPtr<T[]>::ForEachThreadUnseq(F&& Func) const
{
    auto Ret = The.GetNew(The.Size());
    std::for_each(
        std::execution::unseq,
        Ret.begin(),
        Ret.end(),
        Func
    );
    return Ret;
}

// ------------------------------------------------------------------------

template<typename T>
CIN void RA::SharedPtr<T[]>::__SetSize__(const xint FnLeng)
{
    MnLeng = FnLeng;
    The.SetIterator(The.Ptr(), &MnLeng);
}

template<typename T>
CIN void RA::SharedPtr<T[]>::__SetInitialized__(const bool FbInit)
{
    The.MbInitialized = FbInit;
}

#endif
