#pragma once

// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Mutex.h"
#include "Thread/MutexHandler.h"

namespace RA
{
#ifndef __ATOMIC_DEFINED__
#define __ATOMIC_DEFINED__
    template<typename T, typename enabler_t = void>
    class Atomic;

    template<typename T> class Atomic<T, typename std::enable_if_t< IsFundamental(T) || IsSimple(T)>>;
    template<typename T> class Atomic<T, typename std::enable_if_t<!IsFundamental(T) && !IsSimple(T)>>;
#endif
};

namespace RA
{
    template<typename T>
    class Atomic<T, typename std::enable_if_t<!IsFundamental(T) && !IsSimple(T)>> : 
        public std::atomic<std::shared_ptr<T>>, public MutexHandler
    {
    public:
        //using std::atomic<T>::atomic;

        constexpr Atomic() noexcept = default;
        constexpr Atomic(std::nullptr_t) noexcept : Atomic() {}

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

        // Copy Data ----------------------------------------------------------------------------------------------
        template<class O> inline typename std::enable_if< IsSame(Atomic<T>, Atomic<O>), void>::type
        /*void*/ operator=(const O& Other) { The.store(Other); }
        template<class O> inline typename std::enable_if<!IsSame(Atomic<T>, Atomic<O>), void>::type
        /*void*/ operator=(const O& Other) { The.store(static_cast<T>(Other)); }
        // Move Data ----------------------------------------------------------------------------------------------
        template<class O> inline typename std::enable_if< IsSame(Atomic<T>, Atomic<O>), void>::type
        /*void*/ operator=(O&& Other) { The.store(std::make_shared<T>(std::move(Other))); }
        template<class O> inline typename std::enable_if<!IsSame(Atomic<T>, Atomic<O>), void>::type
        /*void*/ operator=(O&& Other) { The.store(std::make_shared<T>(std::move(static_cast<T>(Other)))); }


        Atomic(const T& Other)  { The.store(std::make_shared<T>(Other)); }
        Atomic(      T&& Other) { The.store(std::make_shared<T>(std::move(Other))); }

        constexpr bool operator< (const RA::Atomic<T>& Other) const { return The.Get() < Other.Get(); }
        constexpr bool operator> (const RA::Atomic<T>& Other) const { return The.Get() > Other.Get(); }
        constexpr bool operator<=(const RA::Atomic<T>& Other) const { return The.Get() <= Other.Get(); }
        constexpr bool operator>=(const RA::Atomic<T>& Other) const { return The.Get() >= Other.Get(); }
        constexpr bool operator==(const RA::Atomic<T>& Other) const { return The.Get() == Other.Get(); }
        constexpr bool operator!=(const RA::Atomic<T>& Other) const { return The.Get() != Other.Get(); }

        constexpr bool operator< (const RA::SharedPtr<T>& Other) const { return The.Get() < Other.Get(); }
        constexpr bool operator> (const RA::SharedPtr<T>& Other) const { return The.Get() > Other.Get(); }
        constexpr bool operator<=(const RA::SharedPtr<T>& Other) const { return The.Get() <= Other.Get(); }
        constexpr bool operator>=(const RA::SharedPtr<T>& Other) const { return The.Get() >= Other.Get(); }
        constexpr bool operator==(const RA::SharedPtr<T>& Other) const { return The.Get() == Other.Get(); }
        constexpr bool operator!=(const RA::SharedPtr<T>& Other) const { return The.Get() != Other.Get(); }

        constexpr bool operator< (const T& Other) const { return The.Get() < Other; }
        constexpr bool operator> (const T& Other) const { return The.Get() > Other; }
        constexpr bool operator<=(const T& Other) const { return The.Get() <= Other; }
        constexpr bool operator>=(const T& Other) const { return The.Get() >= Other; }
        constexpr bool operator==(const T& Other) const { return The.Get() == Other; }
        constexpr bool operator!=(const T& Other) const { return The.Get() != Other; }

        bool operator==(nullptr_t) const
        {
            if (The.load().get() == nullptr)
                return true;
            return false;
        }

        void Set(T&& Other) 
        {
            auto Lock = The.GetMutex().CreateLock();
            The = std::move(Other); 
        }

        void Set(const T&  Other) 
        {
            auto Lock = The.GetMutex().CreateLock();
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