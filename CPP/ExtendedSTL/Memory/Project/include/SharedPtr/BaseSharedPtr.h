#pragma once

#include <memory>

#include "RawMapping.h"

// Concept to check if a type is a smart pointer
template <typename T>
concept ConXPointer = requires(T t) {
    { *t } -> std::same_as<typename T::element_type&>;
};

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

        CIN  BaseSharedPtr() noexcept = default;
        CIN  BaseSharedPtr(nullptr_t) noexcept {} // construct empty SharedPtr

        CIN  BaseSharedPtr(const   BaseSharedPtr<X>& Other) : std::shared_ptr<X>(Other) {}
        CIN  BaseSharedPtr(const std::shared_ptr<X>& Other) : std::shared_ptr<X>(Other) {}

        CIN  BaseSharedPtr(  BaseSharedPtr<X>&& Other) noexcept : std::shared_ptr<X>(std::move(Other)) { Other.MbDestructorSet = false;}
        CIN  BaseSharedPtr(std::shared_ptr<X>&& Other) noexcept : std::shared_ptr<X>(std::move(Other)) {}

        CIN std::shared_ptr<X> GetBase();
        CIN std::shared_ptr<X> SPtr();

        CIN bool operator!(void) const;
        CIN bool IsNull() const;

        CIN void operator=(nullptr_t) { The.reset(); }

        CIN       X& operator*();
        CIN const X& operator*() const;

        CIN       X* operator->();
        CIN const X* operator->() const;

        CIN       X& Get();
        CIN const X& Get() const;

        CIN void Swap(SharedPtr<X>&& Other);
        CIN void Swap(const SharedPtr<X>& Other);
    };
}


template<typename T>
CIN bool RA::BaseSharedPtr<T>::operator!(void) const { return bool(The.get() == nullptr); }

template<typename T>
CIN bool RA::BaseSharedPtr<T>::IsNull() const { return The.get() == nullptr; }

template<typename X>
CIN X& RA::BaseSharedPtr<X>::operator*()
{
    if (The.get() == nullptr)
        throw "Null SharedPtr::operator*()!";
    return *The.get();
}

template<typename X>
CIN const X& RA::BaseSharedPtr<X>::operator*() const
{
    if (The.get() == nullptr)
        throw "Null SharedPtr::operator*()!";
    return *The.get();
}

template<typename X>
CIN X* RA::BaseSharedPtr<X>::operator->()
{
    if (The.get() == nullptr)
        throw "Null SharedPtr::operator->()!";
    return The.get();
}

template<typename X>
CIN const X* RA::BaseSharedPtr<X>::operator->() const
{
    if (The.get() == nullptr)
        throw "Null SharedPtr::operator->()!";
    return The.get();
}

template<typename X>
CIN X& RA::BaseSharedPtr<X>::Get()
{
    if (The.get() == nullptr)
        throw "Null SharedPtr::Get()!";
    return *The.get();
}

template<typename X>
CIN const X& RA::BaseSharedPtr<X>::Get() const
{
    if (The.get() == nullptr)
        throw "Null SharedPtr::Get()!";
    return *The.get();
}

template<typename T>
CIN std::shared_ptr<T> RA::BaseSharedPtr<T>::GetBase()
{
    return The;
}

template<typename T>
CIN std::shared_ptr<T> RA::BaseSharedPtr<T>::SPtr()
{
    return The;
}

template<typename X>
CIN void RA::BaseSharedPtr<X>::Swap(SharedPtr<X>&& Other)
{
    Other.MbDestructorSet = false; // we don't want to destroy what we are about to take
    The._Swap(std::move(Other)); // The gets destroyed and takes Other
}

template<typename X>
CIN void RA::BaseSharedPtr<X>::Swap(const SharedPtr<X>& Other)
{
    The._Swap(Other);
}
