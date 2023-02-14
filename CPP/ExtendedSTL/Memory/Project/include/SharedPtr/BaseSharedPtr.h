#pragma once

#include <memory>

#include "RawMapping.h"

namespace RA
{
    template<typename T> class SharedPtr;
#if _HAS_CXX20
#include <concepts>
    template<typename T> class SharedPtr<T[]>;
#else
    template<typename T> class SharedPtr<T*>;
#endif

    template<typename X>
    class BaseSharedPtr : public std::shared_ptr<X>
    {
    protected:
        bool MbDestructorSet = true;
    public:
        using std::shared_ptr<X>::shared_ptr;
        using std::shared_ptr<X>::operator=;

        constexpr BaseSharedPtr() noexcept = default;
        constexpr BaseSharedPtr(nullptr_t) noexcept {} // construct empty SharedPtr

        inline BaseSharedPtr(const BaseSharedPtr<X>& Other) : std::shared_ptr<X>(Other) {}
        inline BaseSharedPtr(const std::shared_ptr<X>& Other) : std::shared_ptr<X>(Other) {}

        inline BaseSharedPtr(BaseSharedPtr<X>&& Other) noexcept : std::shared_ptr<X>(std::move(Other)) {}
        inline BaseSharedPtr(std::shared_ptr<X>&& Other) noexcept : std::shared_ptr<X>(std::move(Other)) {}

        inline std::shared_ptr<X> GetBase();
        inline std::shared_ptr<X> SPtr();

        inline bool operator!(void) const;
        inline bool IsNull() const;

        inline void operator=(nullptr_t) { The.reset(); }

        inline       X& operator*();
        inline const X& operator*() const;

        inline       X* operator->();
        inline const X* operator->() const;

        inline       X& Get();
        inline const X& Get() const;

        inline void Swap(SharedPtr<X>&& Other);
        inline void Swap(const SharedPtr<X>& Other);
    };
}


template<typename T>
bool RA::BaseSharedPtr<T>::operator!(void) const { return bool(The.get() == nullptr); }

template<typename T>
bool RA::BaseSharedPtr<T>::IsNull() const { return The.get() == nullptr; }

template<typename X>
inline X& RA::BaseSharedPtr<X>::operator*()
{
    if (The.get() == nullptr)
        throw "Null SharedPtr::operator*()!";
    return *The.get();
}

template<typename X>
inline const X& RA::BaseSharedPtr<X>::operator*() const
{
    if (The.get() == nullptr)
        throw "Null SharedPtr::operator*()!";
    return *The.get();
}

template<typename X>
inline X* RA::BaseSharedPtr<X>::operator->()
{
    if (The.get() == nullptr)
        throw "Null SharedPtr::operator->()!";
    return The.get();
}

template<typename X>
inline const X* RA::BaseSharedPtr<X>::operator->() const
{
    if (The.get() == nullptr)
        throw "Null SharedPtr::operator->()!";
    return The.get();
}

template<typename X>
inline X& RA::BaseSharedPtr<X>::Get()
{
    if (The.get() == nullptr)
        throw "Null SharedPtr::Get()!";
    return *The.get();
}

template<typename X>
inline const X& RA::BaseSharedPtr<X>::Get() const
{
    if (The.get() == nullptr)
        throw "Null SharedPtr::Get()!";
    return *The.get();
}

template<typename T>
inline std::shared_ptr<T> RA::BaseSharedPtr<T>::GetBase()
{
    return The;
}

template<typename T>
inline std::shared_ptr<T> RA::BaseSharedPtr<T>::SPtr()
{
    return The;
}

template<typename X>
inline void RA::BaseSharedPtr<X>::Swap(SharedPtr<X>&& Other)
{
    Other.MbDestructorSet = false; // we don't want to destroy what we are about to take
    The._Swap(std::move(Other)); // The gets destroyed and takes Other
}

template<typename X>
inline void RA::BaseSharedPtr<X>::Swap(const SharedPtr<X>& Other)
{
    The._Swap(Other);
}