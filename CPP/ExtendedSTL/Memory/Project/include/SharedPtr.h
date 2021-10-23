#pragma once

// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include <memory>

#ifndef __XPTR__
#define __XPTR__
#include <memory>
template<class T>
using xptr = std::shared_ptr<T>;
#define MakePtr std::make_shared
#endif

#ifndef __THE__
#define The (*this)
#endif

namespace RA
{
    template<typename T>
    class SharedPtr : public std::shared_ptr<T>
    {
    public:
        constexpr SharedPtr() noexcept = default;
        constexpr SharedPtr(nullptr_t) noexcept {} // construct empty SharedPtr

        SharedPtr(const SharedPtr<T>&        Other) : std::shared_ptr<T>(Other) {}
        SharedPtr(const std::shared_ptr<T>&  Other) : std::shared_ptr<T>(Other) {}

        SharedPtr(      SharedPtr<T>&&       Other) noexcept : std::shared_ptr<T>(std::move(Other)) {}
        SharedPtr(      std::shared_ptr<T>&& Other) noexcept : std::shared_ptr<T>(std::move(Other)) {}

        //SharedPtr(const T&   Other) : std::shared_ptr<T>(MakePtr<T>(Other)) {}
        //SharedPtr(      T&&  Other) : std::shared_ptr<T>(MakePtr<T>(std::move(Other))) {}

        // Deleted Functions
        //void operator=(const T&  Other) { The.reset(new T(Other)); }
        //void operator=(      T&& Other) { The.reset(new T(std::move(Other))); }

        // ---------------------------------------------------------------------------------------

        SharedPtr& operator=(const SharedPtr<T>& _Right) noexcept;

        template <typename T2>
        SharedPtr& operator=(const SharedPtr<T2>& _Right) noexcept;

        SharedPtr& operator=(SharedPtr<T>&& _Right) noexcept;

        template <typename T2>
        SharedPtr& operator=(SharedPtr<T2>&& _Right) noexcept;
        // ---------------------------------------------------------------------------------------

        SharedPtr& operator=(const std::shared_ptr<T>& _Right) noexcept;

        template <typename T2>
        SharedPtr& operator=(const std::shared_ptr<T2>& _Right) noexcept;

        SharedPtr& operator=(std::shared_ptr<T>&& _Right) noexcept;

        template <typename T2>
        SharedPtr& operator=(std::shared_ptr<T2>&& _Right) noexcept;

        bool operator!() const;
        bool IsNull() const;

        T& Get();

        const T& Get() const;

        T* Ptr();

        const T* Ptr() const;

        std::shared_ptr<T> GetBase();

        std::shared_ptr<T> SPtr();
    };
};

template<typename T>
RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(const RA::SharedPtr<T>& _Right) noexcept {
    SharedPtr(_Right).swap(*this);
    return *this;
}

template<typename T>
template<typename T2>
RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(const RA::SharedPtr<T2>& _Right) noexcept {
    SharedPtr(_Right).swap(*this);
    return *this;
}

template<typename T>
RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(RA::SharedPtr<T>&& _Right) noexcept { // take resource from _Right
    SharedPtr(_STD move(_Right)).swap(*this);
    return *this;
}

template<typename T>
template<typename T2>
RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(RA::SharedPtr<T2>&& _Right) noexcept { // take resource from _Right
    SharedPtr(_STD move(_Right)).swap(*this);
    return *this;
}
// ---------------------------------------------------------------------------------------

template<typename T>
RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(const std::shared_ptr<T>& _Right) noexcept {
    SharedPtr(_Right).swap(*this);
    return *this;
}

template<typename T>
template<typename T2>
RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(const std::shared_ptr<T2>& _Right) noexcept {
    SharedPtr(_Right).swap(*this);
    return *this;
}

template<typename T>
RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(std::shared_ptr<T>&& _Right) noexcept { // take resource from _Right
    SharedPtr(_STD move(_Right)).swap(*this);
    return *this;
}

template<typename T>
template<typename T2>
RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(std::shared_ptr<T2>&& _Right) noexcept { // take resource from _Right
    SharedPtr(_STD move(_Right)).swap(*this);
    return *this;
}

template<typename T>
bool RA::SharedPtr<T>::operator!() const { return bool(The.get() == nullptr); }


template<typename T>
bool RA::SharedPtr<T>::IsNull() const { return The.get() == nullptr; }

template<typename T>
T& RA::SharedPtr<T>::Get()
{
    if (The.get() == nullptr)
        throw "Null SharedPtr::Get()!";
    return *The.get();
}

template<typename T>
const T& RA::SharedPtr<T>::Get() const
{
    if (The.get() == nullptr)
        throw "Null SharedPtr::Get()!";
    return *The.get();
}

template<typename T>
T* RA::SharedPtr<T>::Ptr()
{
    if (The.get() == nullptr)
        throw "Null SharedPtr::Ptr()!";
    return The.get();
}

template<typename T>
const T* RA::SharedPtr<T>::Ptr() const
{
    if (The.get() == nullptr)
        throw "Null SharedPtr::Ptr()!";
    return The.get();
}

template<typename T>
std::shared_ptr<T> RA::SharedPtr<T>::GetBase()
{
    return The;
}

template<typename T>
std::shared_ptr<T> RA::SharedPtr<T>::SPtr()
{
    return The;
}


