#include "RA_Mutex.h"

size_t RA::Mutex::Mutex_Total = 0;


RA::Mutex::Mutex() {
    Mutex_Total++;
    id = Mutex_Total; 
    // first mutex starts at 1 because 0 indicates no mutex
};

void RA::Mutex::WaitAndLock()
{
    std::unique_lock<std::mutex> Lock(key);
    sig.wait(Lock, [this]() { return !MbMutexLocked; });
    SetLockOn();
}

void RA::Mutex::SetLockOn()
{
    MbUseMutex    = true; 
    MbMutexLocked = true;
}

void RA::Mutex::SetLockOff()
{
    MbMutexLocked = false; 
    sig.notify_one();
}

void RA::Mutex::SetMutexOn(){
    MbUseMutex = true;
}

void RA::Mutex::SetMutexOff(){
    MbUseMutex = false;
    MbMutexLocked = false;
    sig.notify_one();
}
