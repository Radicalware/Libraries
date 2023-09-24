#pragma once

#include <memory>

#include "SharedPtr/BaseSharedPtr.h"

namespace RA
{
    template<typename T>
    class SharedPtr : public BaseSharedPtr<T>
    {
    public:
        using BaseSharedPtr<T>::BaseSharedPtr;
        using BaseSharedPtr<T>::operator=;

        // -------------------------------------------------------------------------------------------------
        CIN SharedPtr() noexcept = default;
        CIN SharedPtr(nullptr_t) noexcept {} // construct empty SharedPtr
        // -------------------------------------------------------------------------------------------------
        CIN SharedPtr(const T&  Other) : BaseSharedPtr<T>(std::make_shared<T>(Other)) {}
        CIN SharedPtr(      T&& Other) : BaseSharedPtr<T>(std::make_shared<T>(std::move(Other))) {}
        // -------------------------------------------------------------------------------------------------
        CIN SharedPtr(const SharedPtr<T>& Other)           : BaseSharedPtr<T>(Other) {}
        CIN SharedPtr(      SharedPtr<T>&& Other) noexcept : BaseSharedPtr<T>(std::move(Other)) {}

        CIN SharedPtr(const std::shared_ptr<T>& Other)           : BaseSharedPtr<T>(Other) {}
        CIN SharedPtr(      std::shared_ptr<T>&& Other) noexcept : BaseSharedPtr<T>(std::move(Other)) {}
        // -------------------------------------------------------------------------------------------------
        CIN SharedPtr& operator=(const SharedPtr<T>&  _Right) noexcept;
        CIN SharedPtr& operator=(      SharedPtr<T>&& _Right) noexcept;

        template <typename T2> CIN SharedPtr& operator=(const SharedPtr<T2>&  _Right) noexcept;
        template <typename T2> CIN SharedPtr& operator=(      SharedPtr<T2>&& _Right) noexcept;
        // -------------------------------------------------------------------------------------------------
        CIN SharedPtr& operator=(const std::shared_ptr<T>&  _Right) noexcept;
        CIN SharedPtr& operator=(      std::shared_ptr<T>&& _Right) noexcept;

        template <typename T2> CIN SharedPtr& operator=(const std::shared_ptr<T2>&  _Right) noexcept;
        template <typename T2> CIN SharedPtr& operator=(      std::shared_ptr<T2>&& _Right) noexcept;
        // -------------------------------------------------------------------------------------------------

        CIN bool operator< (const SharedPtr& Other) const; // Used for std::sort
        CIN bool operator> (const SharedPtr& Other) const;
        CIN bool operator<=(const SharedPtr& Other) const;
        CIN bool operator>=(const SharedPtr& Other) const;
        CIN bool operator==(const SharedPtr& Other) const;
        CIN bool operator!=(const SharedPtr& Other) const;

        CIN bool operator< (const T& Other) const; // Used for std::sort
        CIN bool operator> (const T& Other) const;
        CIN bool operator<=(const T& Other) const;
        CIN bool operator>=(const T& Other) const;
        CIN bool operator==(const T& Other) const;
        CIN bool operator!=(const T& Other) const;

        CIN bool operator!=(nullptr_t) const;
        CIN bool operator==(nullptr_t) const;

        //SharedPtr(const T&   Other) : std::shared_ptr<T>(MakePtr<T>(Other)) {}
        //SharedPtr(      T&&  Other) : std::shared_ptr<T>(MakePtr<T>(std::move(Other))) {}

        // Deleted Functions
        //void operator=(const T&  Other) { The.reset(new T(Other)); }
        //void operator=(      T&& Other) { The.reset(new T(std::move(Other))); }

        CIN void Clone(const SharedPtr<T>& Other);
        CIN void Clone(SharedPtr<T>&& Other);

        CIN void Clone(const T& Other);
        CIN void Clone(T&& Other);

        _NODISCARD CIN       std::remove_extent_t<T>* Ptr()       noexcept { return The.get(); }
        _NODISCARD CIN const std::remove_extent_t<T>* Ptr() const noexcept { return The.get(); }

        _NODISCARD CIN       std::remove_extent_t<T>* Raw()       noexcept { return The.get(); }
        _NODISCARD CIN const std::remove_extent_t<T>* Raw() const noexcept { return The.get(); }

        // CIN xint Size() const { throw "RA::SharedPtr<T>::Size >> Don't Call The Function"; }
    };
};



template<typename T>
CIN bool RA::SharedPtr<T>::operator<(const RA::SharedPtr<T>& Other) const {
    return The.Get() < Other.Get();
}
template<typename T>
CIN bool RA::SharedPtr<T>::operator>(const RA::SharedPtr<T>& Other) const {
    return The.Get() > Other.Get();
}
template<typename T>
CIN bool RA::SharedPtr<T>::operator<=(const RA::SharedPtr<T>& Other) const {
    return The.Get() <= Other.Get();
}
template<typename T>
CIN bool RA::SharedPtr<T>::operator>=(const RA::SharedPtr<T>& Other) const {
    return The.Get() >= Other.Get();
}
template<typename T>
CIN bool RA::SharedPtr<T>::operator==(const RA::SharedPtr<T>& Other) const {
    return The.Get() == Other.Get();
}
template<typename T>
CIN bool RA::SharedPtr<T>::operator!=(const RA::SharedPtr<T>& Other) const {
    return The.Get() != Other.Get();
}

template<typename T>
CIN bool RA::SharedPtr<T>::operator<(const T& Other) const {
    return The.Get() < Other;
}
template<typename T>
CIN bool RA::SharedPtr<T>::operator>(const T& Other) const {
    return The.Get() > Other;
}
template<typename T>
CIN bool RA::SharedPtr<T>::operator<=(const T& Other) const {
    return The.Get() <= Other;
}
template<typename T>
CIN bool RA::SharedPtr<T>::operator>=(const T& Other) const {
    return The.Get() >= Other;
}
template<typename T>
CIN bool RA::SharedPtr<T>::operator==(const T& Other) const {
    return The.Get() == Other;
}
template<typename T>
CIN bool RA::SharedPtr<T>::operator!=(const T& Other) const {
    return The.Get() != Other;
}

template<typename T>
CIN bool RA::SharedPtr<T>::operator!=(nullptr_t) const {
    return The.get() != nullptr;
}
template<typename T>
CIN bool RA::SharedPtr<T>::operator==(nullptr_t) const {
    return The.get() == nullptr;
}

// ---------------------------------------------------------------------------------------
template<typename T>
CIN RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(const RA::SharedPtr<T>& _Right) noexcept {
    RA::SharedPtr(_Right).swap(*this);
    return *this;
}

template<typename T>
CIN RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(RA::SharedPtr<T>&& _Right) noexcept { // take resource from _Right
    RA::SharedPtr(_STD move(_Right)).swap(*this);
    return *this;
}

template<typename T>
template<typename T2>
CIN RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(const RA::SharedPtr<T2>& _Right) noexcept {
    RA::SharedPtr(_Right).swap(*this);
    return *this;
}

template<typename T>
template<typename T2>
CIN RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(RA::SharedPtr<T2>&& _Right) noexcept { // take resource from _Right
    RA::SharedPtr(_STD move(_Right)).swap(*this);
    return *this;
}
// ---------------------------------------------------------------------------------------
template<typename T>
CIN RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(const std::shared_ptr<T>& _Right) noexcept {
    RA::SharedPtr(_Right).swap(*this);
    return *this;
}

template<typename T>
CIN RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(std::shared_ptr<T>&& _Right) noexcept { // take resource from _Right
    RA::SharedPtr(_STD move(_Right)).swap(*this);
    return *this;
}

template<typename T>
template<typename T2>
CIN RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(const std::shared_ptr<T2>& _Right) noexcept {
    RA::SharedPtr(_Right).swap(*this);
    return *this;
}

template<typename T>
template<typename T2>
CIN RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(std::shared_ptr<T2>&& _Right) noexcept { // take resource from _Right
    RA::SharedPtr(_STD move(_Right)).swap(*this);
    return *this;
}
// ---------------------------------------------------------------------------------------

template<typename T>
CIN void RA::SharedPtr<T>::Clone(const RA::SharedPtr<T>& Other)
{
    if (!Other)
        return;
    The = RA::SharedPtr<T>(Other.Get());
}

template<typename T>
CIN void RA::SharedPtr<T>::Clone(RA::SharedPtr<T>&& Other)
{
    The = std::move(Other);
}

template<typename T>
CIN void RA::SharedPtr<T>::Clone(const T& Other)
{
    The = RA::SharedPtr<T>(Other);
}

template<typename T>
CIN void RA::SharedPtr<T>::Clone(T&& Other)
{
    The = RA::SharedPtr<T>(std::move(Other));
}
