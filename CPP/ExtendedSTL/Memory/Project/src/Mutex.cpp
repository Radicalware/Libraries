#include "Mutex.h"

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
#include<Windows.h>
#else 
#include<unistd.h>
#endif

RA::Mutex::Mutex() 
{
    id = MutexTotal;
    MutexTotal++;
    if (MutexTotal > 65530)
        MutexTotal = 1;
};

RA::Lock RA::Mutex::CreateLock()
{
    auto LoLock = std::make_unique<std::unique_lock<std::mutex>>(MoKeyPtr.Get());
    MoSigPtr.Get().wait(*LoLock.get(), [this]() { return (!MbUseMutex) || (!MbMutexLockedPtr.Get()); });
    return RA::Lock(std::move(LoLock), MoKeyPtr, MoSigPtr, MbMutexLockedPtr);
}

void RA::Mutex::Wait() const
{
    while (MbMutexLockedPtr.Get() == true && MbUseMutex == true)
    {
        #if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
                ::Sleep(1);
        #else
                ::usleep(1);
        #endif
    }
}

void RA::Mutex::Unlock()
{
    MbMutexLockedPtr.Get() = false;
    MoSigPtr.Get().notify_one();
}

void RA::Mutex::UnlockAll()
{
    MbMutexLockedPtr.Get() = false;
    MoSigPtr.Get().notify_all();
}

void RA::Mutex::SignalOne()
{
    Unlock();
}

void RA::Mutex::SignalAll()
{
    UnlockAll();
}

void RA::Mutex::SetMutexOn(){
    MbUseMutex = true;
}

void RA::Mutex::SetMutexOff(){
    if (!MbMutexLockedPtr.Get())
        return;
    MbUseMutex = false;
    MbMutexLockedPtr.Get() = false;
    MoSigPtr.Get().notify_all();
}
