#pragma once

// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include <memory>

#ifndef __THE__
#define __THE__
#define The  (*this)
#endif

namespace RA
{
    template<typename T>
    class SharedPtr;

    template <typename T, typename ...A>
    SharedPtr<T> MakeShared(A&&... Args);

    template <typename T>
    SharedPtr<T[]> MakeSharedArray(const size_t FnSize);

    template<typename T>
    class SharedPtr : public std::shared_ptr<T>
    {
    public:
        constexpr SharedPtr() noexcept = default;
        constexpr SharedPtr(nullptr_t) noexcept {} // construct empty SharedPtr
        using std::shared_ptr<T>::shared_ptr;

        SharedPtr(const SharedPtr<T>&        Other) : std::shared_ptr<T>(Other) {}
        SharedPtr(const std::shared_ptr<T>&  Other) : std::shared_ptr<T>(Other) {}

        SharedPtr(      SharedPtr<T>&&       Other) noexcept : std::shared_ptr<T>(std::move(Other)) {}
        SharedPtr(      std::shared_ptr<T>&& Other) noexcept : std::shared_ptr<T>(std::move(Other)) {}

        inline void operator=(nullptr_t) { The.reset(); }

        bool operator< (const RA::SharedPtr<T>& Other) const; // Used for std::sort
        bool operator> (const RA::SharedPtr<T>& Other) const;
        bool operator<=(const RA::SharedPtr<T>& Other) const;
        bool operator>=(const RA::SharedPtr<T>& Other) const;
        bool operator==(const RA::SharedPtr<T>& Other) const;
        bool operator!=(const RA::SharedPtr<T>& Other) const;

        bool operator< (const T& Other) const; // Used for std::sort
        bool operator> (const T& Other) const;
        bool operator<=(const T& Other) const;
        bool operator>=(const T& Other) const;
        bool operator==(const T& Other) const;
        bool operator!=(const T& Other) const;

        bool operator!=(nullptr_t) const;
        bool operator==(nullptr_t) const;

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

        bool operator!(void) const;
        bool IsNull() const;

        
              T& operator*();
        const T& operator*() const;

              T& Get();
        const T& Get() const;

        _NODISCARD       std::remove_extent_t<T>* Ptr()       noexcept { return The.get(); }
        _NODISCARD const std::remove_extent_t<T>* Ptr() const noexcept { return The.get(); }

        _NODISCARD       std::remove_extent_t<T>* Raw()       noexcept { return The.get(); }
        _NODISCARD const std::remove_extent_t<T>* Raw() const noexcept { return The.get(); }

        void Clone(const RA::SharedPtr<T>&  Other);
        void Clone(      RA::SharedPtr<T>&& Other);

        void Clone(const T&  Other);
        void Clone(      T&& Other);

        std::shared_ptr<T> GetBase();
        std::shared_ptr<T> SPtr();
    };
};

template<typename T>
bool RA::SharedPtr<T>::operator<(const RA::SharedPtr<T>& Other) const {
    return The.Get() < Other.Get();
}
template<typename T>
bool RA::SharedPtr<T>::operator>(const RA::SharedPtr<T>& Other) const {
    return The.Get() > Other.Get();
}
template<typename T>
bool RA::SharedPtr<T>::operator<=(const RA::SharedPtr<T>& Other) const {
    return The.Get() <= Other.Get();
}
template<typename T>
bool RA::SharedPtr<T>::operator>=(const RA::SharedPtr<T>& Other) const {
    return The.Get() >= Other.Get();
}
template<typename T>
bool RA::SharedPtr<T>::operator==(const RA::SharedPtr<T>& Other) const {
    return The.Get() == Other.Get();
}
template<typename T>
bool RA::SharedPtr<T>::operator!=(const RA::SharedPtr<T>& Other) const {
    return The.Get() != Other.Get();
}

template<typename T>
bool RA::SharedPtr<T>::operator<(const T& Other) const {
    return The.Get() < Other;
}
template<typename T>
bool RA::SharedPtr<T>::operator>(const T& Other) const {
    return The.Get() > Other;
}
template<typename T>
bool RA::SharedPtr<T>::operator<=(const T& Other) const {
    return The.Get() <= Other;
}
template<typename T>
bool RA::SharedPtr<T>::operator>=(const T& Other) const {
    return The.Get() >= Other;
}
template<typename T>
bool RA::SharedPtr<T>::operator==(const T& Other) const {
    return The.Get() == Other;
}
template<typename T>
bool RA::SharedPtr<T>::operator!=(const T& Other) const {
    return The.Get() != Other;
}

template<typename T>
bool RA::SharedPtr<T>::operator!=(nullptr_t) const {
    return The.get() != nullptr;
}
template<typename T>
bool RA::SharedPtr<T>::operator==(nullptr_t) const {
    return The.get() == nullptr;
}


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
bool RA::SharedPtr<T>::operator!(void) const { return bool(The.get() == nullptr); }


template<typename T>
bool RA::SharedPtr<T>::IsNull() const { return The.get() == nullptr; }

template<typename T>
T& RA::SharedPtr<T>::operator*()
{
    if (The.get() == nullptr)
        throw "Null SharedPtr::operator*()!";
    return *The.get();
}

template<typename T>
const T& RA::SharedPtr<T>::operator*() const
{
    if (The.get() == nullptr)
        throw "Null SharedPtr::operator*()!";
    return *The.get();
}

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
void RA::SharedPtr<T>::Clone(const RA::SharedPtr<T>& Other)
{
    if (Other.get() == nullptr)
        return;
    The = RA::MakeShared<T>(*Other.get());
}

template<typename T>
inline void RA::SharedPtr<T>::Clone(RA::SharedPtr<T>&& Other)
{
    The = std::move(Other);
}

template<typename T>
void RA::SharedPtr<T>::Clone(const T& Other)
{
    The = RA::MakeShared<T>(Other);
}

template<typename T>
inline void RA::SharedPtr<T>::Clone(T&& Other)
{
    The = RA::MakeShared<T>(std::move(Other));
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

namespace RA
{
    template <typename T, typename ...A> RA::SharedPtr<T> MakeShared(A&&... Args) 
    { 
        return std::make_shared<T>(std::forward<A>(Args)...); 
    }

    template <typename T>
    RA::SharedPtr<T[]> MakeSharedArray(const size_t FnSize)
    {
        auto SPtr = RA::SharedPtr<T[]>(new T[FnSize + 1]);
        for (auto* ptr = SPtr.Raw(); ptr < SPtr.Raw() + FnSize + 1; ptr++)
            *ptr = '\0';
        return SPtr;
    }
}

#ifndef __XPTR__
#define __XPTR__

template<class T>
using xptr = RA::SharedPtr<T>;

template<class T>
using xp = RA::SharedPtr<T>;

template<class T>
using sxp = std::shared_ptr<T>;

#define  MKP RA::MakeShared
#define MMKP RA::MoveMakeShared
#define MKPA RA::MakeSharedArray

#define MakePtr std::make_shared
#endif

namespace RA
{
    template <class T> struct IsTypeSharedPtr : std::false_type {};
    template <class T> struct IsTypeSharedPtr<RA::SharedPtr<T>> : std::true_type {};
}