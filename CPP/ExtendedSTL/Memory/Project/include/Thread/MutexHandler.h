#pragma once

#include <unordered_map>

#include "SharedPtr.h"
#include "Mutex.h"

namespace RA
{
    class MutexHandler
    {
    public:
        RIN constexpr MutexHandler() noexcept {}
        RA::Mutex&    ReadyMutex();
        bool          UsingMutex() const;
        RA::Mutex&    GetMutex()   const;
        RIN auto      NewLock()       { return ReadyMutex().NewLock(); }
        RIN auto      NewLock() const { return GetMutex().NewLock(); }
        RIN void      Unlock()        { return ReadyMutex().Unlock(); }
        RIN void      Unlock()  const { return GetMutex().Unlock(); }

    protected:
                xint MnMutexID = 0;
        istatic xint SnMutexCount = 0;
        istatic std::unordered_map<xint, RA::Mutex> SmMutex;
        istatic std::mutex SoMutex;
    };
};
