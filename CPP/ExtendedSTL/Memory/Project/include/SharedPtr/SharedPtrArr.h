#pragma once

#include "SharedPtr/BaseSharedPtr.h"
#include "Iterator.h"

#include <functional>

namespace RA
{
    template<typename T>
    class SharedPtr<T*> : public BaseSharedPtr<T>, public RA::Iterator<T>
    {
        size_t MnSize;
        void SetSize(const size_t FnSize);
        std::function<void(T*)> MfDestructor;
        bool MbUsingCustomDestructor = false;
    public:
        ~SharedPtr();
        template<typename F> inline SharedPtr<T>& SetDestructor(F&& FfDestructor);

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
        The.MfDestructor(The.Ptr());
}

template<typename T>
template<typename F>
inline RA::SharedPtr<T>& RA::SharedPtr<T*>::SetDestructor(F&& FfDestructor)
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
}

template<typename T>
template<typename ...A, class TT, typename std::enable_if< IsClass(TT), bool>::type>
inline RA::SharedPtr<T*>::SharedPtr(const size_t FnSize, A&&... Args) :
    std::shared_ptr<T>(new T[FnSize](std::forward<A>(Args)...), [](T* FtPtr) { delete[] FtPtr; })
{
    The.SetSize(FnSize);
}

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
