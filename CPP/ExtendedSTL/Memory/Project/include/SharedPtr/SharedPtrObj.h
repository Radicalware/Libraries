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
        constexpr SharedPtr() noexcept = default;
        constexpr SharedPtr(nullptr_t) noexcept {} // construct empty SharedPtr
        // -------------------------------------------------------------------------------------------------
        inline SharedPtr(const T&  Other) : BaseSharedPtr<T>(std::make_shared<T>(Other)) {}
        inline SharedPtr(      T&& Other) : BaseSharedPtr<T>(std::make_shared<T>(std::move(Other))) {}
        // -------------------------------------------------------------------------------------------------
        inline SharedPtr(const SharedPtr<T>& Other)           : BaseSharedPtr<T>(Other) {}
        inline SharedPtr(      SharedPtr<T>&& Other) noexcept : BaseSharedPtr<T>(std::move(Other)) {}

        inline SharedPtr(const std::shared_ptr<T>& Other)           : BaseSharedPtr<T>(Other) {}
        inline SharedPtr(      std::shared_ptr<T>&& Other) noexcept : BaseSharedPtr<T>(std::move(Other)) {}
        // -------------------------------------------------------------------------------------------------
        inline SharedPtr& operator=(const SharedPtr<T>&  _Right) noexcept;
        inline SharedPtr& operator=(      SharedPtr<T>&& _Right) noexcept;

        template <typename T2> inline SharedPtr& operator=(const SharedPtr<T2>&  _Right) noexcept;
        template <typename T2> inline SharedPtr& operator=(      SharedPtr<T2>&& _Right) noexcept;
        // -------------------------------------------------------------------------------------------------
        inline SharedPtr& operator=(const std::shared_ptr<T>&  _Right) noexcept;
        inline SharedPtr& operator=(      std::shared_ptr<T>&& _Right) noexcept;

        template <typename T2> inline SharedPtr& operator=(const std::shared_ptr<T2>&  _Right) noexcept;
        template <typename T2> inline SharedPtr& operator=(      std::shared_ptr<T2>&& _Right) noexcept;
        // -------------------------------------------------------------------------------------------------

        inline bool operator< (const SharedPtr& Other) const; // Used for std::sort
        inline bool operator> (const SharedPtr& Other) const;
        inline bool operator<=(const SharedPtr& Other) const;
        inline bool operator>=(const SharedPtr& Other) const;
        inline bool operator==(const SharedPtr& Other) const;
        inline bool operator!=(const SharedPtr& Other) const;

        inline bool operator< (const T& Other) const; // Used for std::sort
        inline bool operator> (const T& Other) const;
        inline bool operator<=(const T& Other) const;
        inline bool operator>=(const T& Other) const;
        inline bool operator==(const T& Other) const;
        inline bool operator!=(const T& Other) const;

        inline bool operator!=(nullptr_t) const;
        inline bool operator==(nullptr_t) const;

        //SharedPtr(const T&   Other) : std::shared_ptr<T>(MakePtr<T>(Other)) {}
        //SharedPtr(      T&&  Other) : std::shared_ptr<T>(MakePtr<T>(std::move(Other))) {}

        // Deleted Functions
        //void operator=(const T&  Other) { The.reset(new T(Other)); }
        //void operator=(      T&& Other) { The.reset(new T(std::move(Other))); }

        inline void Clone(const SharedPtr<T>& Other);
        inline void Clone(SharedPtr<T>&& Other);

        inline void Clone(const T& Other);
        inline void Clone(T&& Other);

        _NODISCARD inline       std::remove_extent_t<T>* Ptr()       noexcept { return The.get(); }
        _NODISCARD inline const std::remove_extent_t<T>* Ptr() const noexcept { return The.get(); }

        _NODISCARD inline       std::remove_extent_t<T>* Raw()       noexcept { return The.get(); }
        _NODISCARD inline const std::remove_extent_t<T>* Raw() const noexcept { return The.get(); }

        // inline xint Size() const { throw "RA::SharedPtr<T>::Size >> Don't Call The Function"; }
    };
};



template<typename T>
inline bool RA::SharedPtr<T>::operator<(const RA::SharedPtr<T>& Other) const {
    return The.Get() < Other.Get();
}
template<typename T>
inline bool RA::SharedPtr<T>::operator>(const RA::SharedPtr<T>& Other) const {
    return The.Get() > Other.Get();
}
template<typename T>
inline bool RA::SharedPtr<T>::operator<=(const RA::SharedPtr<T>& Other) const {
    return The.Get() <= Other.Get();
}
template<typename T>
inline bool RA::SharedPtr<T>::operator>=(const RA::SharedPtr<T>& Other) const {
    return The.Get() >= Other.Get();
}
template<typename T>
inline bool RA::SharedPtr<T>::operator==(const RA::SharedPtr<T>& Other) const {
    return The.Get() == Other.Get();
}
template<typename T>
inline bool RA::SharedPtr<T>::operator!=(const RA::SharedPtr<T>& Other) const {
    return The.Get() != Other.Get();
}

template<typename T>
inline bool RA::SharedPtr<T>::operator<(const T& Other) const {
    return The.Get() < Other;
}
template<typename T>
inline bool RA::SharedPtr<T>::operator>(const T& Other) const {
    return The.Get() > Other;
}
template<typename T>
inline bool RA::SharedPtr<T>::operator<=(const T& Other) const {
    return The.Get() <= Other;
}
template<typename T>
inline bool RA::SharedPtr<T>::operator>=(const T& Other) const {
    return The.Get() >= Other;
}
template<typename T>
inline bool RA::SharedPtr<T>::operator==(const T& Other) const {
    return The.Get() == Other;
}
template<typename T>
inline bool RA::SharedPtr<T>::operator!=(const T& Other) const {
    return The.Get() != Other;
}

template<typename T>
inline bool RA::SharedPtr<T>::operator!=(nullptr_t) const {
    return The.get() != nullptr;
}
template<typename T>
inline bool RA::SharedPtr<T>::operator==(nullptr_t) const {
    return The.get() == nullptr;
}

// ---------------------------------------------------------------------------------------
template<typename T>
inline RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(const RA::SharedPtr<T>& _Right) noexcept {
    RA::SharedPtr(_Right).swap(*this);
    return *this;
}

template<typename T>
inline RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(RA::SharedPtr<T>&& _Right) noexcept { // take resource from _Right
    RA::SharedPtr(_STD move(_Right)).swap(*this);
    return *this;
}

template<typename T>
template<typename T2>
inline RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(const RA::SharedPtr<T2>& _Right) noexcept {
    RA::SharedPtr(_Right).swap(*this);
    return *this;
}

template<typename T>
template<typename T2>
inline RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(RA::SharedPtr<T2>&& _Right) noexcept { // take resource from _Right
    RA::SharedPtr(_STD move(_Right)).swap(*this);
    return *this;
}
// ---------------------------------------------------------------------------------------
template<typename T>
inline RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(const std::shared_ptr<T>& _Right) noexcept {
    RA::SharedPtr(_Right).swap(*this);
    return *this;
}

template<typename T>
inline RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(std::shared_ptr<T>&& _Right) noexcept { // take resource from _Right
    RA::SharedPtr(_STD move(_Right)).swap(*this);
    return *this;
}

template<typename T>
template<typename T2>
inline RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(const std::shared_ptr<T2>& _Right) noexcept {
    RA::SharedPtr(_Right).swap(*this);
    return *this;
}

template<typename T>
template<typename T2>
inline RA::SharedPtr<T>& RA::SharedPtr<T>::operator=(std::shared_ptr<T2>&& _Right) noexcept { // take resource from _Right
    RA::SharedPtr(_STD move(_Right)).swap(*this);
    return *this;
}
// ---------------------------------------------------------------------------------------

template<typename T>
inline void RA::SharedPtr<T>::Clone(const RA::SharedPtr<T>& Other)
{
    if (!Other)
        return;
    The = RA::SharedPtr<T>(Other.Get());
}

template<typename T>
inline void RA::SharedPtr<T>::Clone(RA::SharedPtr<T>&& Other)
{
    The = std::move(Other);
}

template<typename T>
inline void RA::SharedPtr<T>::Clone(const T& Other)
{
    The = RA::SharedPtr<T>(Other);
}

template<typename T>
inline void RA::SharedPtr<T>::Clone(T&& Other)
{
    The = RA::SharedPtr<T>(std::move(Other));
}
