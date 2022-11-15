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
        short MbDestructorCount = 0;
    public:
        ~SharedPtr();

        template<typename F, typename ...A>
        SharedPtr<T*>& Initialize(F&& FfInitialize, A&&... Args);
        SharedPtr<T*>& Destroy(std::function<void(T&)>&& FfDestructor);

        BaseSharedPtr<T>::BaseSharedPtr;

        template<class TT = T, typename std::enable_if<!IsClass(TT), bool>::type = 0>
        inline SharedPtr(const size_t FnSize);
        template<typename ...A, class TT = T, typename std::enable_if< IsClass(TT), bool>::type = 0>
        inline SharedPtr(const size_t FnSize, A&&... Args);

        inline SharedPtr(SharedPtr<T*>&& Other);
        inline SharedPtr(const SharedPtr<T*>& Other);

        inline void operator=(SharedPtr<T*>&& Other);
        inline void operator=(const SharedPtr<T*>& Other);

        inline void Clone(SharedPtr<T*>&& Other);
        inline void Clone(const SharedPtr<T*>& Other);

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
    if (MbDestructorCount >= 2)
        for (auto& Obj : The)
            The.MfDestructor(Obj);
    MbDestructorCount++;
}

template<typename T>
template<typename F, typename ...A>
inline RA::SharedPtr<T*>& RA::SharedPtr<T*>::Initialize(F&& FfInitialize, A&&... Args)
{
    if (MbInitialized)
        throw "RA::SharedPtr<T*>::Initialize >> Already Initialized!";
    MbInitialized = true;
    for (auto& Obj : The)
        FfInitialize(Obj, std::forward<A>(Args)...);
    return The;
}

template<typename T>
inline RA::SharedPtr<T*>& RA::SharedPtr<T*>::Destroy(std::function<void(T&)>&& FfDestructor)
{
    if (MbDestructorCount)
        throw "RA::SharedPtr Destructor Set";
    MbDestructorCount++;
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
    RA::BaseSharedPtr<T>(new T[FnSize](std::forward<A>(Args)...), [](T* FtPtr) { delete[] FtPtr; })
{
    The.SetSize(FnSize);
}
#else
template<typename T>
template<typename ...A, class TT, typename std::enable_if< IsClass(TT), bool>::type>
inline RA::SharedPtr<T*>::SharedPtr(const size_t FnSize, A&&... Args) :
    RA::BaseSharedPtr<T>(new T[FnSize], [](T* FtPtr) { delete[] FtPtr; })
{
    The.SetSize(FnSize);
}
#endif

template<typename T>
inline RA::SharedPtr<T*>::SharedPtr(RA::SharedPtr<T*>&& Other)
{
    std::shared_ptr(std::move(Other)).swap(The);
    The.SetSize(Other.Size());
    MbDestructorCount = Other.MbDestructorCount;
    MfDestructor = std::move(Other.MfDestructor);
}

template<typename T>
inline RA::SharedPtr<T*>::SharedPtr(const RA::SharedPtr<T*>& Other)
{
    std::shared_ptr(Other).swap(The);
    The.SetSize(Other.Size());
    MbDestructorCount = Other.MbDestructorCount;
    MfDestructor = Other.MfDestructor;
}

template<typename T>
inline void RA::SharedPtr<T*>::operator=(RA::SharedPtr<T*>&& Other)
{
    std::shared_ptr(std::move(Other)).swap(The);
    The.SetSize(Other.Size());
    MbDestructorCount = Other.MbDestructorCount;
    MfDestructor = std::move(Other.MfDestructor);
}

template<typename T>
inline void RA::SharedPtr<T*>::operator=(const RA::SharedPtr<T*>& Other)
{
    std::shared_ptr(Other).swap(The);
    The.SetSize(Other.Size());
    MbDestructorCount = Other.MbDestructorCount;
    MfDestructor = Other.MfDestructor;
}

template<typename T>
inline void RA::SharedPtr<T*>::Clone(SharedPtr<T*>&& Other)
{
    The = std::move(Other);
}

template<typename T>
inline void RA::SharedPtr<T*>::Clone(const SharedPtr<T*>& Other)
{
    if (!Other)
    {
        The = nullptr;
        MnSize = 0;
        return;
    }
    The = RA::MakeShared<T*>(Other.Size());
    MbDestructorCount = 0;
    MfDestructor = Other.MfDestructor;
    if (Other.MnSize)
        memcpy(The.Ptr(), Other.Ptr(), Other.Size() * sizeof(T) + sizeof(uint));
}
