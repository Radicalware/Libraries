#pragma once

// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "SharedPtr.h"
#include "Lock.h"
#include <atomic>

#include<mutex>
#include<condition_variable>


namespace RA
{
    // THIS SHOULD ONLY BE USED IN A SHARED POINTER WHEN BEING SENT TO NEXUS
    class Mutex // used to lock/unlock objects on a given index
    {
        RA::SharedPtr<std::atomic<xint>>        id = RA::MakeShared<std::atomic<xint>>(1);
        RA::SharedPtr<std::condition_variable>  MoSigPtr = RA::MakeShared<std::condition_variable>();
        RA::SharedPtr<std::mutex>               MoKeyPtr = RA::MakeShared<std::mutex>(); // key for a lock (unique_lock)
        RA::SharedPtr<std::atomic<bool>>        MbMutexLockedPtr = RA::MakeShared<std::atomic<bool>>(false);
        RA::SharedPtr<std::atomic<bool>>        MbUseMtx = RA::MakeShared<std::atomic<bool>>(true);
        static std::atomic<xint> MutexTotal; // 1 because 0 designates don't ues Mutex to Nexus

    public:
        Mutex();

        size_t        GetID() const { return *id; }
        static size_t GetMutexTotal() { return MutexTotal; }
        static void   ResetTotalMutexCount() { MutexTotal = 0; }


        template<typename F>
        INL auto CreateLock(F&& Function);
        //    RA::Lock CreateLock();
        INL auto CreateLock() { return std::unique_lock<std::mutex>(MoKeyPtr.Get()); }
            void     Wait() const;

        void     Unlock();
        void     UnlockAll();
        void     SignalOne();
        void     SignalAll();
        
        void     SetUseMutexOff()   { *MbUseMtx = false;}
        void     SetUseMutexOn()    { *MbUseMtx = true;}
        bool     BxUseMutex() const { return *MbUseMtx; }

        bool     BxLocked()   const { return   *MbMutexLockedPtr;  }
        bool     BxUnlocked() const { return !(*MbMutexLockedPtr); }
    };
};


template<typename F>
INL auto RA::Mutex::CreateLock(F&& Function)
{
    try
    {
        auto LoLock = std::make_unique<std::unique_lock<std::mutex>>(MoKeyPtr.Get());
        MoSigPtr.Get().wait(*LoLock.get(), [this, &Function]()
            { return (!MbMutexLockedPtr.Get() && Function()); });
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