﻿#include "Thread/Lock.h"

RA::Lock::Lock(
    RA::SharedPtr<std::unique_lock<std::mutex>> FoLock,
    RA::SharedPtr<std::mutex>& FoKey,
    RA::SharedPtr<std::condition_variable>& FoCon,
    RA::SharedPtr<std::atomic<bool>>& FbMutexLockedPtr)
    :   
        MoLockPtr(std::move(FoLock)), 
        MoKeyPtr(FoKey), 
        MoConditionalVarPtr(FoCon), 
        MbMutexLockedPtr(FbMutexLockedPtr)
{
    MbMutexLockedPtr.Get() = true;
}

RA::Lock::Lock(const Lock& Other)
{
    The = Other;
}

RA::Lock::Lock(Lock&& Other) noexcept
{
    The = std::move(Other);
}

void RA::Lock::operator=(const Lock& Other)
{
    MoLockPtr = Other.MoLockPtr;
    MoKeyPtr = Other.MoKeyPtr;
    MoConditionalVarPtr = Other.MoConditionalVarPtr;
    MbMutexLockedPtr = Other.MbMutexLockedPtr;
}

void RA::Lock::operator=(Lock&& Other) noexcept
{
    MoLockPtr = std::move(Other.MoLockPtr);
    MoKeyPtr = std::move(Other.MoKeyPtr);
    MoConditionalVarPtr = std::move(Other.MoConditionalVarPtr);
    MbMutexLockedPtr = std::move(Other.MbMutexLockedPtr);
}

RA::Lock::~Lock()
{
    MbMutexLockedPtr.Get() = false;
    MoLockPtr.Get().unlock();
    MoLockPtr.Get().release();
    MoConditionalVarPtr.Get().notify_one();
}
