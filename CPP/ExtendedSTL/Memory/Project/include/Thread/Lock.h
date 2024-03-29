﻿#pragma once

// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include <memory>
#include <mutex>

#include "SharedPtr.h"


namespace RA
{
    class Lock
    {
    public:
        Lock(
            RA::SharedPtr<std::unique_lock<std::mutex>> FoLock,
            RA::SharedPtr<std::mutex>& FoKey, 
            RA::SharedPtr<std::condition_variable>& FoCon,
            RA::SharedPtr<std::atomic<bool>>& FbMutexLockedPtr);

        Lock(const Lock& Other);
        Lock(Lock&& Other) noexcept;

        void operator=(const Lock& Other);
        void operator=(Lock&& Other) noexcept;

        ~Lock();
    private:
        RA::SharedPtr<std::unique_lock<std::mutex>> MoLockPtr;
        RA::SharedPtr<std::mutex>                   MoKeyPtr;
        RA::SharedPtr<std::condition_variable>      MoConditionalVarPtr;
        RA::SharedPtr<std::atomic<bool>>            MbMutexLockedPtr;
    };
}
