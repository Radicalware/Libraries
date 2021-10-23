#include "Lock.h"

RA::Lock::Lock(
    std::unique_ptr<std::unique_lock<std::mutex>> FoLock,
    RA::SharedPtr<std::mutex>& FoKey,
    RA::SharedPtr<std::condition_variable>& FoCon,
    RA::SharedPtr<std::atomic<bool>>& FbMutexLockedPtr)
    :   
        MoLock(std::move(FoLock)), 
        MoKeyPtr(FoKey), 
        MoConditionalVarPtr(FoCon), 
        MbMutexLockedPtr(FbMutexLockedPtr)
{
    MbMutexLockedPtr.Get() = true;
}

RA::Lock::~Lock()
{
    MbMutexLockedPtr.Get() = false;
    MoLock.get()->unlock();
    MoLock.release();
    MoConditionalVarPtr.Get().notify_one();
}
