#pragma once

#include<mutex>
#include<condition_variable>


#ifndef __XPTR__
#define __XPTR__
#include <memory>
template<class T>
using xptr = std::shared_ptr<T>;
#define MakePtr std::make_shared
#endif

template<typename T> class Nexus;

namespace RA
{
    // THIS SHOULD ONLY BE USED IN AN XPTR
    class Mutex // used to lock/unlock objects on a given index
    {
        friend class Nexus<void>;

        size_t id = 0;
        std::mutex key; // key for a lock (unique_lock)
        std::condition_variable sig;
        bool MbMutexLocked = false; // Is the asset locked?
        bool MbUseMutex    = true;      // Should we ignore the mutex?

        static size_t Mutex_Total; // = 0 

    public:
        Mutex();

        void WaitAndLock();
        void SetLockOn();
        void SetLockOff();

        // Is mutex being ignored or not?
        bool MutexIsOn()  const { return  MbUseMutex; }
        bool MutexIsOff() const { return !MbUseMutex; }

        // Is mutex turned on?
        void SetMutexOn();
        void SetMutexOff();
    };
};