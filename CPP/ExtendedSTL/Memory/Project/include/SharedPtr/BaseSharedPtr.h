#pragma once

#include <memory>

#include "RawMapping.h"

namespace RA
{
    template<typename T> class SharedPtr;
    template<typename T> class SharedPtr<T*>;

    template <typename T, typename ...A>
    _NODISCARD std::enable_if_t<!IsArray(T) && !IsPointer(T), SharedPtr<T>>
        inline MakeShared(A&&... Args);

    template <typename PA>
    _NODISCARD std::enable_if_t<IsPointer(PA) && !IsClass(RemovePtr(PA)), SharedPtr<PA>>
        inline MakeShared(const uint FnLeng);

    template <typename PA, typename ...A>
    _NODISCARD std::enable_if_t<IsPointer(PA) && IsClass(RemovePtr(PA)), SharedPtr<PA>>
        inline MakeShared(const uint FnLeng, A&&... Args);

    template <typename T>
    inline SharedPtr<T*>
        MakeSharedBuffer(const uint FnLeng, const uint FnUnitByteSize);
    template <typename T, typename D>
    inline SharedPtr<T*>
        MakeSharedBuffer(const uint FnLeng, const uint FnUnitByteSize, D&& FfDestructor);
}

namespace RA
{
    template<typename X>
    class BaseSharedPtr : public std::shared_ptr<X>
    {
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

        inline X& operator*();
        inline const X& operator*() const;

        inline X& Get();
        inline const X& Get() const;
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
