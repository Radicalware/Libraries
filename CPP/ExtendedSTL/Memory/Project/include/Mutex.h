#pragma once

// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Lock.h"

#include<mutex>
#include<condition_variable>


namespace RA
{
    // THIS SHOULD ONLY BE USED IN A SHARED POINTER WHEN BEING SENT TO NEXUS
    class Mutex // used to lock/unlock objects on a given index
    {
        size_t                                  id = 1; // 1 because 0 designates don't ues Mutex to Nexus
        RA::SharedPtr<std::condition_variable>  MoSigPtr = MakePtr<std::condition_variable>();
        RA::SharedPtr<std::mutex>               MoKeyPtr = MakePtr<std::mutex>(); // key for a lock (unique_lock)
        RA::SharedPtr<std::atomic<bool>>        MbMutexLockedPtr = MakePtr<std::atomic<bool>>(false);
        bool                                    MbUseMutex = true; // Should we ignore the mutex?

        inline static std::atomic<size_t>       MutexTotal = 1; // 1 because 0 designates don't ues Mutex to Nexus

    public:
        Mutex();

        size_t        GetID() const { return id; }
        static size_t GetMutexTotal() { return MutexTotal; }
        static void   ResetTotalMutexCount() { MutexTotal = 0; }

        template<typename F>
        RA::Lock CreateLock(F&& Function);
        RA::Lock CreateLock();
        void     Wait() const;

        bool     IsLocked()   const { return  MbMutexLockedPtr.Get(); }
        bool     IsUnlocked() const { return !MbMutexLockedPtr.Get(); }

        void     Unlock();
        void     UnlockAll();
        void     SignalOne();
        void     SignalAll();

        // Is mutex being ignored or not?
        bool     IsMutexOn()  const { return  MbUseMutex; }
        bool     IsMutexOff() const { return !MbUseMutex; }

        // Is mutex turned on?
        void     SetMutexOn();
        void     SetMutexOff();
    };
};


template<typename F>
inline RA::Lock RA::Mutex::CreateLock(F&& Function)
{
    try
    {
        auto LoLock = std::make_unique<std::unique_lock<std::mutex>>(MoKeyPtr.Get());
        MoSigPtr.Get().wait(*LoLock.get(), [this, &Function]()
            { return ((!MbUseMutex) || (!MbMutexLockedPtr.Get())) && Function(); });
        MbMutexLockedPtr.Get() = true;
        return RA::Lock(std::move(LoLock), MoKeyPtr, MoSigPtr, MbMutexLockedPtr);
    }
    catch (const char* Err)
    {
        throw Err;
    }
    catch (const std::exception& Err)
    {
        throw Err.what();
    }

}