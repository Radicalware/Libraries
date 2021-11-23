#pragma once

// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Atomic/BaseAtomic.h"

namespace RA
{
#ifndef __ATOMIC_DEFINED__
#define __ATOMIC_DEFINED__
    template<typename T, typename enabler_t = void> class Atomic;
#endif

    template<typename T>
    class Atomic<T, typename std::enable_if_t<std::is_class<T>::value>> : public BaseAtomic<std::shared_ptr<T>>
    {
    public:
        using BaseAtomic<std::shared_ptr<T>>::BaseAtomic;

        inline std::atomic<std::shared_ptr<T>>& Base() const { return *reinterpret_cast<std::atomic<std::shared_ptr<T>>>(this); }

        constexpr Atomic() noexcept {};
        constexpr Atomic(nullptr_t) noexcept {}

             // Copy Reference To Data >> RA Smart Pointer
             Atomic(   const SharedPtr<T>&  Other){ The.store(Other); }
        void operator=(const SharedPtr<T>&  Other){ The.store(Other); }
             Atomic(         SharedPtr<T>&& Other){ The.store(std::move(Other)); }
        void operator=(      SharedPtr<T>&& Other){ The.store(std::move(Other)); }

             // Copy Reference To Data >> std Smart Pointer
             Atomic(   const std::shared_ptr<T>&  Other){ The.store(Other); }
        void operator=(const std::shared_ptr<T>&  Other){ The.store(Other); }
             Atomic(         std::shared_ptr<T>&& Other){ The.store(std::move(Other)); }
        void operator=(      std::shared_ptr<T>&& Other){ The.store(std::move(Other)); }

             // Copy RA::Atomic<T>
             Atomic(   const RA::Atomic<T>&  Other) { The.store(Other.Ptr()); }
        void operator=(const RA::Atomic<T>&  Other) { The.store(Other.Ptr()); }
             Atomic(         RA::Atomic<T>&& Other) { The.store(std::move(Other.Ptr())); }
        void operator=(      RA::Atomic<T>&& Other) { The.store(std::move(Other.Ptr())); }

             // Deep copy data
             Atomic(   const T& Other){ The.store(MakePtr<T>(Other)); }
        inline typename std::enable_if<!std::is_same<T, Atomic<T>>::value, void>::type
             operator=(const T& Other){ The.store(MakePtr<T>(Other)); }

             // Move Data
             Atomic(         T&& Other) { The.store(MakePtr<T>(std::move(Other))); }
        inline typename std::enable_if<!std::is_same<T, Atomic<T>>::value, void>::type
             operator=(      T&& Other) { The.store(MakePtr<T>(std::move(Other))); }


        bool operator< (const RA::Atomic<T>& Other) const { return The.Get() < Other.Get(); }
        bool operator> (const RA::Atomic<T>& Other) const { return The.Get() > Other.Get(); }
        bool operator<=(const RA::Atomic<T>& Other) const { return The.Get() <= Other.Get(); }
        bool operator>=(const RA::Atomic<T>& Other) const { return The.Get() >= Other.Get(); }
        bool operator==(const RA::Atomic<T>& Other) const { return The.Get() == Other.Get(); }
        bool operator!=(const RA::Atomic<T>& Other) const { return The.Get() != Other.Get(); }

        bool operator< (const RA::SharedPtr<T>& Other) const { return The.Get() < Other.Get(); }
        bool operator> (const RA::SharedPtr<T>& Other) const { return The.Get() > Other.Get(); }
        bool operator<=(const RA::SharedPtr<T>& Other) const { return The.Get() <= Other.Get(); }
        bool operator>=(const RA::SharedPtr<T>& Other) const { return The.Get() >= Other.Get(); }
        bool operator==(const RA::SharedPtr<T>& Other) const { return The.Get() == Other.Get(); }
        bool operator!=(const RA::SharedPtr<T>& Other) const { return The.Get() != Other.Get(); }

        bool operator< (const T& Other) const { return The.Get() < Other; }
        bool operator> (const T& Other) const { return The.Get() > Other; }
        bool operator<=(const T& Other) const { return The.Get() <= Other; }
        bool operator>=(const T& Other) const { return The.Get() >= Other; }
        bool operator==(const T& Other) const { return The.Get() == Other; }
        bool operator!=(const T& Other) const { return The.Get() != Other; }

        bool operator==(nullptr_t) const
        {
            if (The.load().get() == nullptr)
                return true;
            return false;
        }

        void Set(T&& Other) 
        {
            GetMutex<T>().CreateLock();
            The = std::move(Other); 
            GetMutex<T>().SetLockOff();
        }

        void Set(const T&  Other) 
        {
            auto Lock = GetMutex<T>().CreateLock();
            The = Other;
        }

        T Get() const
        {
            if (The.load() == nullptr)
                throw "Null aptr::Get()!";
            return *The.load().get();
            // .load = creates copy of SPtr
            // .get  = gets T* of the SharedPtr
            // *     = goes from T* to T
            // This Get is actually a reference the the same
            // object called multiple times as the SharedPtr
            // address is shared accross multiple threads.
        }

        RA::SharedPtr<T>                Ptr()       const { return The.load(); }
        std::atomic<std::shared_ptr<T>> AtomicPtr() const { return The; }
    };
}


template<class T>
using ap = RA::Atomic<T>;