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

             Atomic() { };
                // RA Smart Pointer
             Atomic(   const SharedPtr<T>&  Other){ The.store(Other); }
        void operator=(const SharedPtr<T>&  Other){ The.store(Other); }
             Atomic(         SharedPtr<T>&& Other){ The.store(std::move(Other)); }
        void operator=(      SharedPtr<T>&& Other){ The.store(std::move(Other)); }
                // std Smart Pointer
             Atomic(   const std::shared_ptr<T>&  Other){ The.store(Other); }
        void operator=(const std::shared_ptr<T>&  Other){ The.store(Other); }
             Atomic(         std::shared_ptr<T>&& Other){ The.store(std::move(Other)); }
        void operator=(      std::shared_ptr<T>&& Other){ The.store(std::move(Other)); }
                // Copy Atomic -- These are deleted functions
             Atomic(   const Atomic<T>&  Other) { The.store(MakePtr<T>(Other.GetCopy())); }
        void operator=(const Atomic<T>&  Other) { The.store(MakePtr<T>(Other.GetCopy())); }
                // Move Atomic
             Atomic(         Atomic<T>&& Other) { The.exchange(Other); }
        void operator=(      Atomic<T>&& Other) { The.exchange(Other); }
                // New Object
             Atomic(   const T& Other){ The.store(MakePtr<T>(Other)); }
        inline typename std::enable_if<!std::is_same<T, Atomic<T>>::value, void>::type
             operator=(const T& Other){ The.store(MakePtr<T>(Other)); }

             Atomic(         T&& Other) { The.store(MakePtr<T>(std::move(Other))); }
        inline typename std::enable_if<!std::is_same<T, Atomic<T>>::value, void>::type
             operator=(      T&& Other) { The.store(MakePtr<T>(std::move(Other))); }

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

        T GetCopy() const
        {
            if (The.load().get() == nullptr)
                throw "Null aptr::Get()!";
            return *The.load().get();
        }

        const T& Get() const
        {
            if (The.load().get() == nullptr)
                throw "Null aptr::Get()!";
            return *The.load().get();
        }

        //T& Get()
        //{
        //    if (The.load().get() == nullptr)
        //        throw "Null aptr::Get()!";
        //    return *The.load().get();
        //}

        const T* const                  Ptr() const       { return The.load().get(); }
        RA::SharedPtr<T>                SPtr() const      { return The.load(); }
        std::atomic<std::shared_ptr<T>> AtomicPtr() const { return The; }
    };
}

