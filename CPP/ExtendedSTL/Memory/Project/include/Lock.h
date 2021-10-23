#pragma once

// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "SharedPtr.h"

#include <memory>
#include <mutex>

namespace RA
{
    class Lock
    {
    public:
        Lock(
            std::unique_ptr<std::unique_lock<std::mutex>> FoLock,
            RA::SharedPtr<std::mutex>& FoKey, 
            RA::SharedPtr<std::condition_variable>& FoCon,
            RA::SharedPtr<std::atomic<bool>>& FbMutexLockedPtr);
        ~Lock();
    private:
        std::unique_ptr<std::unique_lock<std::mutex>> MoLock;
        RA::SharedPtr<std::mutex>                     MoKeyPtr;
        RA::SharedPtr<std::condition_variable>        MoConditionalVarPtr;
        RA::SharedPtr<std::atomic<bool>>              MbMutexLockedPtr;
    };
}
