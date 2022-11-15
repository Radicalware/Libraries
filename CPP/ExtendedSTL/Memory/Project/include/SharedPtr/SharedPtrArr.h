#pragma once

#include "SharedPtr/BaseSharedPtr.h"
#include "Iterator.h"

#include <functional>
#include <utility>

namespace RA
{
    template<typename T>
    class SharedPtr<T*> : public BaseSharedPtr<T>, public RA::Iterator<T>
    {
        size_t MnSize;
        void SetSize(const size_t FnSize);
        std::function<void(T&)> MfDestructor;
        bool MbInitialized = false;
        bool MbUsingCustomDestructor = false;
    public:
        ~SharedPtr();
        SharedPtr<T*>& Initialize(std::function<void(T&)>&& FfInitialize);
        SharedPtr<T*>& Destroy(std::function<void(T&)>&& FfDestructor);

        BaseSharedPtr<T>::BaseSharedPtr;

        template<class TT = T, typename std::enable_if<!IsClass(TT), bool>::type = 0>
        inline SharedPtr(const size_t FnSize);
        template<typename ...A, class TT = T, typename std::enable_if< IsClass(TT), bool>::type = 0>
        inline SharedPtr(const size_t FnSize, A&&... Args);

        inline SharedPtr(SharedPtr<T>&& Other);
        inline SharedPtr(const SharedPtr<T>& Other);

        inline void operator=(SharedPtr<T>&& Other);
        inline void operator=(const SharedPtr<T>& Other);

        _NODISCARD inline       T* Ptr()       noexcept { return The.get(); }
        _NODISCARD inline const T* Ptr() const noexcept { return The.get(); }

        _NODISCARD inline       T* Raw()       noexcept { return The.get(); }
        _NODISCARD inline const T* Raw() const noexcept { return The.get(); }

        inline        T& operator[](const size_t Idx)       noexcept { return The.get()[Idx]; }
        inline  const T& operator[](const size_t Idx) const noexcept { return The.get()[Idx]; }

        constexpr auto Size() const { return MnSize; }
    };
}


template<typename T>
inline void RA::SharedPtr<T*>::SetSize(const size_t FnSize)
{
    MnSize = FnSize;
    The.SetIterator(The.Ptr(), &MnSize);
}

template<typename T>
inline RA::SharedPtr<T*>::~SharedPtr()
{
    if (MbUsingCustomDestructor)
        for (auto& Obj : The)
            The.MfDestructor(Obj);
}

template<typename T>
inline RA::SharedPtr<T*>& RA::SharedPtr<T*>::Initialize(std::function<void(T&)>&& FfInitialize)
{
    if (MbInitialized)
        throw "RA::SharedPtr<T*>::Initialize >> Already Initialized!";
    MbInitialized = true;
    for (auto& Obj : The)
        FfInitialize(Obj);
}

template<typename T>
inline RA::SharedPtr<T*>& RA::SharedPtr<T*>::Destroy(std::function<void(T&)>&& FfDestructor)
{
    MbUsingCustomDestructor = true;
    The.MfDestructor = FfDestructor;
    return The;
}

template<typename T>
template<class TT, typename std::enable_if<!IsClass(TT), bool>::type>
inline RA::SharedPtr<T*>::SharedPtr(const size_t FnSize) :
    RA::BaseSharedPtr<T>(new T[FnSize+1], [](T* FtPtr) { delete[] FtPtr; })
{
    The.SetSize(FnSize);
    for (auto* ptr = The.get(); ptr < The.get() + FnSize + 1; ptr++)
        *ptr = 0;
}

#ifndef UsingNVCC
template<typename T>
template<typename ...A, class TT, typename std::enable_if< IsClass(TT), bool>::type>
inline RA::SharedPtr<T*>::SharedPtr(const size_t FnSize, A&&... Args) :
    RA::BaseSharedPtr<T>(new T[FnSize](Args...), [](T* FtPtr) { delete[] FtPtr; })
{
    The.SetSize(FnSize);
}
#else
template<typename T>
template<typename ...A, class TT, typename std::enable_if< IsClass(TT), bool>::type>
inline RA::SharedPtr<T*>::SharedPtr(const size_t FnSize, A&&... Args) :
    RA::BaseSharedPtr<T>(new T[FnSize](std::forward<A>(Args)...), [](T* FtPtr) { delete[] FtPtr; })
{
    The.SetSize(FnSize);
}
#endif

template<typename T>
inline RA::SharedPtr<T*>::SharedPtr(RA::SharedPtr<T>&& Other)
{
    RA::SharedPtr(std::move(Other)).swap(The);
    The.SetSize(Other.Size());
    MbUsingCustomDestructor = true;
    MfDestructor = std::move(Other.MfDestructor);
}

template<typename T>
inline RA::SharedPtr<T*>::SharedPtr(const RA::SharedPtr<T>& Other)
{
    RA::SharedPtr(Other).swap(The);
    The.SetSize(Other.Size());
    MbUsingCustomDestructor = true;
    MfDestructor = Other.MfDestructor;
}

template<typename T>
inline void RA::SharedPtr<T*>::operator=(RA::SharedPtr<T>&& Other)
{
    RA::SharedPtr<T>(std::move(Other)).swap(The);
    The.SetSize(Other.Size());
    MbUsingCustomDestructor = true;
    MfDestructor = std::move(Other.MfDestructor);
}

template<typename T>
inline void RA::SharedPtr<T*>::operator=(const RA::SharedPtr<T>& Other)
{
    RA::SharedPtr<T>(Other).swap(The);
    The.SetSize(Other.Size());
    MbUsingCustomDestructor = true;
    MfDestructor = Other.MfDestructor;
}
