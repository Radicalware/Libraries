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
        size_t MnLeng = 0;
        size_t MnUnitSize = 0;
        std::function<void(T&)> MfDestructor;
        bool MbInitialized = false;
        bool MbDestructorSet = false;

    public:
        ~SharedPtr();

        template<typename F, typename ...A>
        SharedPtr<T*> Initialize(F&& FfInitialize, A&&... Args);
        SharedPtr<T*> SetDestructor(std::function<void(T&)>&& FfDestructor);

        BaseSharedPtr<T>::BaseSharedPtr;

        template<class TT = T, typename std::enable_if<!IsClass(TT), bool>::type = 0>
        inline SharedPtr(const size_t FnLeng);
        template<typename ...A, class TT = T, typename std::enable_if< IsClass(TT), bool>::type = 0>
        inline SharedPtr(const size_t FnLeng, A&&... Args);

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

        constexpr auto Size()   const { return MnLeng; }
        constexpr auto GetLength() const { return MnLeng; }
        constexpr auto GetUnitSize() const { return MnUnitSize; }
        constexpr auto GetAllocationSize() const { return MnUnitSize * MnLeng + sizeof(uint); }

        void __SetSize__(const size_t FnLeng); // todo: get friend functions working
    };
}


template<typename T>
inline void RA::SharedPtr<T*>::__SetSize__(const size_t FnLeng)
{
    MnLeng = FnLeng;
    The.SetIterator(The.Ptr(), &MnLeng);
}

template<typename T>
inline RA::SharedPtr<T*>::~SharedPtr()
{
    if (MbDestructorSet && The.use_count() == 1)
        for (auto& Obj : The)
            The.MfDestructor(Obj);
}

template<typename T>
template<typename F, typename ...A>
inline RA::SharedPtr<T*> RA::SharedPtr<T*>::Initialize(F&& FfInitialize, A&&... Args)
{
    if (MbInitialized)
        throw "RA::SharedPtr<T*>::Initialize >> Already Initialized!";
    MbInitialized = true;
    for (auto& Obj : The)
        FfInitialize(Obj, std::forward<A>(Args)...);
    return The;
}

template<typename T>
inline RA::SharedPtr<T*> RA::SharedPtr<T*>::SetDestructor(std::function<void(T&)>&& FfDestructor)
{
    if (MbDestructorSet)
        throw "RA::SharedPtr Destructor Set";
    MbDestructorSet = true;
    The.MfDestructor = FfDestructor;
    return The;
}

template<typename T>
template<class TT, typename std::enable_if<!IsClass(TT), bool>::type>
inline RA::SharedPtr<T*>::SharedPtr(const size_t FnLeng) :
    RA::BaseSharedPtr<T>(new T[FnLeng + 1], [](T* FtPtr) { delete[] FtPtr; })
{
    The.__SetSize__(FnLeng);
    for (auto* ptr = The.get(); ptr < The.get() + FnLeng + 1; ptr++)
        *ptr = 0;
}

#ifndef UsingNVCC
template<typename T>
template<typename ...A, class TT, typename std::enable_if< IsClass(TT), bool>::type>
inline RA::SharedPtr<T*>::SharedPtr(const size_t FnLeng, A&&... Args) :
    RA::BaseSharedPtr<T>(new T[FnLeng](std::forward<A>(Args)...), [](T* FtPtr) { delete[] FtPtr; })
{
    The.__SetSize__(FnLeng);
}
#else
template<typename T>
template<typename ...A, class TT, typename std::enable_if< IsClass(TT), bool>::type>
inline RA::SharedPtr<T*>::SharedPtr(const size_t FnLeng, A&&... Args) :
    RA::BaseSharedPtr<T>(new T[FnLeng], [](T* FtPtr) { delete[] FtPtr; })
{
    The.__SetSize__(FnLeng);
}
#endif

template<typename T>
inline RA::SharedPtr<T*>::SharedPtr(RA::SharedPtr<T*>&& Other)
{
    std::shared_ptr(std::move(Other)).swap(The);
    The.__SetSize__(Other.Size());
    MbDestructorSet = Other.MbDestructorSet;
    Other.MbDestructorSet = false; // don't destroy before a move
    MbInitialized = Other.MbInitialized;
    MfDestructor = Other.MfDestructor;
}

template<typename T>
inline RA::SharedPtr<T*>::SharedPtr(const RA::SharedPtr<T*>& Other)
{
    std::shared_ptr(Other).swap(The);
    The.__SetSize__(Other.Size());
    MbDestructorSet = Other.MbDestructorSet;
    MbInitialized = Other.MbInitialized;
    MfDestructor = Other.MfDestructor;
}

template<typename T>
inline void RA::SharedPtr<T*>::operator=(RA::SharedPtr<T*>&& Other)
{
    std::shared_ptr(std::move(Other)).swap(The);
    The.__SetSize__(Other.Size());
    MbDestructorSet = Other.MbDestructorSet;
    MbInitialized = Other.MbInitialized;
    MfDestructor = Other.MfDestructor;
}

template<typename T>
inline void RA::SharedPtr<T*>::operator=(const RA::SharedPtr<T*>& Other)
{
    std::shared_ptr(Other).swap(The);
    The.__SetSize__(Other.Size());
    MbDestructorSet = Other.MbDestructorSet;
    MbInitialized = Other.MbInitialized;
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
        MnLeng = 0;
        return;
    }
    The = RA::MakeShared<T*>(Other.Size());
    MfDestructor = Other.MfDestructor;
    if (Other.MnLeng)
        memcpy(The.Ptr(), Other.Ptr(), Other.Size() * sizeof(T) + sizeof(uint));
    MbDestructorSet = false;
}
