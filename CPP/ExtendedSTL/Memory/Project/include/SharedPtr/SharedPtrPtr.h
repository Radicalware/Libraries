#pragma once

#include "SharedPtr/BaseSharedPtr.h"

#if _HAS_CXX20 == 0
#pragma once

#include "SharedPtr/BaseSharedPtr.h"
#include "Iterator.h"

#include <functional>
#include <algorithm>
#include <utility>

namespace RA
{
    // in C++17, shared_ptr<T> is the same as shared_ptr<T>
    // you know if it's an array by its input (is the input a ptr?)
    // if the input is a pointer, then the type is an array
    // C++20 onwards uses shared_ptr<T*> to remove confusion
    template<typename T>
    class SharedPtr<T*> : public BaseSharedPtr<T*>, public RA::Iterator<T>
    {
    protected:
        xint MnLeng = 0;
        bool MbInitialized = false;
        T* MPtr = nullptr;
    public:
        BaseSharedPtr<T*>::BaseSharedPtr;

        ~SharedPtr();

        template<class TT = T, typename std::enable_if<IsFundamental(TT), bool>::type = 0>
        inline SharedPtr(const xint FnLeng);

        template<typename ...A, class TT = T, typename std::enable_if<!IsFundamental(TT), bool>::type = 0>
        inline SharedPtr(const xint FnLeng, A&&... Args);

        inline SharedPtr(SharedPtr<T*>&& Other);
        inline SharedPtr(const SharedPtr<T*>& Other);

        inline void operator=(SharedPtr<T*>&& Other);
        inline void operator=(const SharedPtr<T*>& Other);

        inline void Clone(SharedPtr<T*>&& Other);
        inline void Clone(const SharedPtr<T*>& Other);

        _NODISCARD inline       T* Ptr()       noexcept { return MPtr; }
        _NODISCARD inline const T* Ptr() const noexcept { return MPtr; }

        _NODISCARD inline       T* Raw()       noexcept { return MPtr; }
        _NODISCARD inline const T* Raw() const noexcept { return MPtr; }

        inline        T& operator[](const xint Idx);
        inline  const T& operator[](const xint Idx) const;

        constexpr auto Size()   const { return MnLeng; }
        constexpr auto GetLength() const { return MnLeng; }
        constexpr auto GetUnitSize() const { return sizeof(T); }
        constexpr auto GetMallocSize() const { return sizeof(T) * MnLeng + sizeof(xint); }

        template<typename F, typename ...A>
        SharedPtr<T*> Proc(F&& Func, A&&... Args);

        template<typename R = T, typename F, typename ...A>
        SharedPtr<R[]> ForEach(F&& Func, A&&... Args);
        template<typename R = T, typename F, typename ...A>
        SharedPtr<R[]> ForEach(F&& Func, A&&... Args) const;

        template<typename R = T, typename F, typename ...A>
        R ForEachAdd(F&& Func, A&&... Args);
        template<typename R = T, typename F, typename ...A>
        R ForEachAdd(F&& Func, A&&... Args) const;

        void __SetSize__(const xint FnLeng); // todo: get friend functions working
        void __SetInitialized__(const bool FbInit);
    };
}

template<typename T>
inline RA::SharedPtr<T*>::~SharedPtr()
{
    if (!The.MbDestructorSet)
        return;
    The.MbDestructorSet = false;
    if (The.MPtr == NULL || The.MPtr == nullptr || !The.MPtr)
        return;
    if (The.MnLeng == 0)
        return;
    if (The.use_count() <= 1)
    {
        if constexpr (!IsFundamental(RemovePtr(T)))
        {
            for (auto& Obj : The)
                Obj.~T();
        }
        DeleteArr(MPtr);
    }
}

template<typename T>
template<class TT, typename std::enable_if<IsFundamental(TT), bool>::type>
inline RA::SharedPtr<T*>::SharedPtr(const xint FnLeng) :
    MPtr(new T[FnLeng + 1]), RA::BaseSharedPtr<T*>(std::make_shared<T*>(MPtr))
{
    The.MbDestructorSet = true;
    The.__SetSize__(FnLeng);
    for (auto* ptr = MPtr; ptr < MPtr + FnLeng + 1; ptr++)
        *ptr = 0;
}

template<typename T>
template<typename ...A, class TT, typename std::enable_if<!IsFundamental(TT), bool>::type>
inline RA::SharedPtr<T*>::SharedPtr(const xint FnLeng, A&&... Args) :
    MPtr(new T[FnLeng]), RA::BaseSharedPtr<T*>(std::make_shared<T*>(MPtr))
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
inline RA::SharedPtr<T*>::SharedPtr(RA::SharedPtr<T*>&& Other)
{
    The.MbDestructorSet = true;
    Other.MbDestructorSet = false;
    The._Move_construct_from(std::move(Other));
    The.MPtr = Other.MPtr;
    The.__SetSize__(Other.Size());
    MbInitialized = Other.MbInitialized;
}

template<typename T>
inline RA::SharedPtr<T*>::SharedPtr(const RA::SharedPtr<T*>& Other)
{
    The.MbDestructorSet = true;
    The._Copy_construct_from(Other);
    The.MPtr = Other.MPtr;
    The.__SetSize__(Other.Size());
    MbInitialized = Other.MbInitialized;
}

template<typename T>
inline void RA::SharedPtr<T*>::operator=(RA::SharedPtr<T*>&& Other)
{
    The.MbDestructorSet = true;
    Other.MbDestructorSet = false;
    The._Move_construct_from(std::move(Other));
    The.MPtr = Other.MPtr;
    The.__SetSize__(Other.Size());
    MbInitialized = Other.MbInitialized;
}

template<typename T>
inline void RA::SharedPtr<T*>::operator=(const RA::SharedPtr<T*>& Other)
{
    The.MbDestructorSet = true;
    The._Copy_construct_from(Other);
    The.MPtr = Other.MPtr;
    The.__SetSize__(Other.Size());
    MbInitialized = Other.MbInitialized;
}

template<typename T>
inline void RA::SharedPtr<T*>::Clone(RA::SharedPtr<T*>&& Other)
{
    The = std::move(Other);
}

template<typename T>
inline void RA::SharedPtr<T*>::Clone(const RA::SharedPtr<T*>& Other)
{
    The.MbDestructorSet = true;
    if (!Other)
    {
        The = nullptr;
        DeleteArr(MPtr);
        MnLeng = 0;
        return;
    }

    DeleteArr(MPtr);
    if (Other.Size())
    {
        if constexpr (IsFundamental(T))
        {
            MPtr = new T[Other.MnLeng + 1];
            memcpy(MPtr, Other.MPtr, Other.MnLeng * sizeof(T));
            MPtr[Other.MnLeng] = 0;
        }
        else
        {
            MPtr = new T[Other.MnLeng];
            for (xint i = 0; i < Other.MnLeng; i++)
                MPtr[i] = Other.MPtr[i];
        }
    }
    The.__SetSize__(Other.Size());
    MbInitialized = Other.MbInitialized;
}

template<typename T>
inline T& RA::SharedPtr<T*>::operator[](const xint Idx)
{
    if (Idx >= MnLeng)
        throw "Idx is out of range!";
    return MPtr[Idx];
}

template<typename T>
inline const T& RA::SharedPtr<T*>::operator[](const xint Idx) const
{
    if (Idx >= MnLeng)
        throw "Idx is out of range!";
    return MPtr[Idx];
}

template<typename T>
template<typename F, typename ...A>
inline RA::SharedPtr<T*> RA::SharedPtr<T*>::Proc(F&& Func, A&& ...Args)
{
    for (auto& Elem : The)
        Func(Elem, std::forward<A>(Args)...);
    return The;
}

template<typename T>
template<typename R, typename F, typename ...A>
inline RA::SharedPtr<R[]> RA::SharedPtr<T*>::ForEach(F&& Func, A&& ...Args)
{
    auto Ret = RA::SharedPtr<T*>(FnLeng, std::forward<T>(Args)...);
    for (xint i = 0; i < MnLeng; i++)
        Ret[i] = Func(MPtr[i], std::forward<A>(Args)...);
    return Ret;
}

template<typename T>
template<typename R, typename F, typename ...A>
inline RA::SharedPtr<R[]> RA::SharedPtr<T*>::ForEach(F&& Func, A && ...Args) const
{
    auto Ret = RA::SharedPtr<T*>(FnLeng, std::forward<T>(Args)...);
    for (xint i = 0; i < MnLeng; i++)
        Ret[i] = Func(MPtr[i], std::forward<A>(Args)...);
    return Ret;
}

template<typename T>
template<typename R, typename F, typename ...A>
inline R RA::SharedPtr<T*>::ForEachAdd(F&& Func, A && ...Args)
{
    R Ret = 0;
    for (xint i = 0; i < MnLeng; i++)
        Ret += Func(MPtr[i], std::forward<A>(Args)...);
    return Ret;
}

template<typename T>
template<typename R, typename F, typename ...A>
inline R RA::SharedPtr<T*>::ForEachAdd(F&& Func, A && ...Args)const
{
    R Ret = 0;
    for (xint i = 0; i < MnLeng; i++)
        Ret += Func(MPtr[i], std::forward<A>(Args)...);
    return Ret;
}

template<typename T>
inline void RA::SharedPtr<T*>::__SetSize__(const xint FnLeng)
{
    MnLeng = FnLeng;
    The.SetIterator(MPtr, &MnLeng);
}

template<typename T>
inline void RA::SharedPtr<T*>::__SetInitialized__(const bool FbInit)
{
    The.MbInitialized = FbInit;
}

#endif // !_HAS_CXX20