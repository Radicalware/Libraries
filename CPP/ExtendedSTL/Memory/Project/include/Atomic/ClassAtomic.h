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

    template<typename T> class Atomic<T, typename std::enable_if_t< BxFundamental(T) || BxSimple(T)>>;
    template<typename T> class Atomic<T, typename std::enable_if_t<!BxFundamental(T) && !BxSimple(T)>>;
#endif
};

namespace RA
{
    template<typename T>
    class Atomic<T, typename std::enable_if_t<!BxFundamental(T) && !BxSimple(T)>> : 
        public std::atomic<std::shared_ptr<T>>, public MutexHandler
    {
    public:
        //using std::atomic<T>::atomic;

        CIN Atomic() noexcept = default;
        CIN Atomic(std::nullptr_t) noexcept : Atomic() {}

             // Copy Reference To Data >> RA Smart Pointer
        CIN      Atomic(   const SharedPtr<T>&  Other){ The.store(Other); }
        CIN void operator=(const SharedPtr<T>&  Other){ The.store(Other); }
                 Atomic(         SharedPtr<T>&& Other){ The.store(std::move(Other)); }
        CIN void operator=(      SharedPtr<T>&& Other){ The.store(std::move(Other)); }

             // Copy Reference To Data >> std Smart Pointer
        CIN      Atomic(   const std::shared_ptr<T>&  Other){ The.store(Other); }
        CIN void operator=(const std::shared_ptr<T>&  Other){ The.store(Other); }
        CIN      Atomic(         std::shared_ptr<T>&& Other){ The.store(std::move(Other)); }
        CIN void operator=(      std::shared_ptr<T>&& Other){ The.store(std::move(Other)); }

             // Copy RA::Atomic<T>
        CIN      Atomic(   const RA::Atomic<T>&  Other) { The.store(Other.Ptr()); }
        CIN void operator=(const RA::Atomic<T>&  Other) { The.store(Other.Ptr()); }
        CIN      Atomic(         RA::Atomic<T>&& Other) { The.store(std::move(Other.Ptr())); }
        CIN void operator=(      RA::Atomic<T>&& Other) { The.store(std::move(Other.Ptr())); }

        // Copy Data ----------------------------------------------------------------------------------------------
        template<class O> CIN typename std::enable_if< BxSameType(Atomic<T>, Atomic<O>), void>::type
        /*void*/ operator=(const O& Other) { The.store(Other); }
        template<class O> CIN typename std::enable_if<!BxSameType(Atomic<T>, Atomic<O>), void>::type
        /*void*/ operator=(const O& Other) { The.store(static_cast<T>(Other)); }
        // Move Data ----------------------------------------------------------------------------------------------
        template<class O> CIN typename std::enable_if< BxSameType(Atomic<T>, Atomic<O>), void>::type
        /*void*/ operator=(O&& Other) { The.store(std::make_shared<T>(std::move(Other))); }
        template<class O> CIN typename std::enable_if<!BxSameType(Atomic<T>, Atomic<O>), void>::type
        /*void*/ operator=(O&& Other) { The.store(std::make_shared<T>(std::move(static_cast<T>(Other)))); }


        CIN Atomic(const T& Other)  { The.store(std::make_shared<T>(Other)); }
        CIN Atomic(      T&& Other) { The.store(std::make_shared<T>(std::move(Other))); }

        CIN  bool operator< (const RA::Atomic<T>& Other) const { return The.Get() < Other.Get(); }
        CIN  bool operator> (const RA::Atomic<T>& Other) const { return The.Get() > Other.Get(); }
        CIN  bool operator<=(const RA::Atomic<T>& Other) const { return The.Get() <= Other.Get(); }
        CIN  bool operator>=(const RA::Atomic<T>& Other) const { return The.Get() >= Other.Get(); }
        CIN  bool operator==(const RA::Atomic<T>& Other) const { return The.Get() == Other.Get(); }
        CIN  bool operator!=(const RA::Atomic<T>& Other) const { return The.Get() != Other.Get(); }

        CIN  bool operator< (const RA::SharedPtr<T>& Other) const { return The.Get() < Other.Get(); }
        CIN  bool operator> (const RA::SharedPtr<T>& Other) const { return The.Get() > Other.Get(); }
        CIN  bool operator<=(const RA::SharedPtr<T>& Other) const { return The.Get() <= Other.Get(); }
        CIN  bool operator>=(const RA::SharedPtr<T>& Other) const { return The.Get() >= Other.Get(); }
        CIN  bool operator==(const RA::SharedPtr<T>& Other) const { return The.Get() == Other.Get(); }
        CIN  bool operator!=(const RA::SharedPtr<T>& Other) const { return The.Get() != Other.Get(); }

        CIN  bool operator< (const T& Other) const { return The.Get() < Other; }
        CIN  bool operator> (const T& Other) const { return The.Get() > Other; }
        CIN  bool operator<=(const T& Other) const { return The.Get() <= Other; }
        CIN  bool operator>=(const T& Other) const { return The.Get() >= Other; }
        CIN  bool operator==(const T& Other) const { return The.Get() == Other; }
        CIN  bool operator!=(const T& Other) const { return The.Get() != Other; }

        CIN bool operator==(nullptr_t) const
        {
            if (The.load().get() == nullptr)
                return true;
            return false;
        }

        CIN void Set(T&& Other)
        {
            auto Lock = The.GetMutex().CreateLock();
            The = std::move(Other); 
        }

        CIN void Set(const T&  Other)
        {
            auto Lock = The.GetMutex().CreateLock();
            The = Other;
        }

        CIN T Get() const
        {
            if (The.load() == nullptr)
                throw "Null aptr::Get()!";
            return *The.load().get();
            // .load = creates copy of SPtr
            // .get  = gets T* of the SharedPtr
            // *     = goes from T* to T
            // The Get is actually a reference the the same
            // object called multiple times as the SharedPtr
            // address is shared accross multiple threads.
        }

        RA::SharedPtr<T>                Ptr()       const { return The.load(); }
        std::atomic<std::shared_ptr<T>> AtomicPtr() const { return The; }
    };
}


template<class T>
using ap = RA::Atomic<T>;
