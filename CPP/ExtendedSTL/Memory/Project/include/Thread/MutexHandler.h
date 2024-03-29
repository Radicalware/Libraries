﻿#pragma once

#include <unordered_map>

#include "SharedPtr.h"
#include "Mutex.h"

namespace RA
{
    class MutexHandler
    {
    public:
        RIN constexpr MutexHandler() noexcept {}
        void          CreateMTX(); // Warning: there is a windows macro "CreateMutex"
        bool          UsingMTX() const;
        RA::Mutex&    GetMTX() const;
        RA::Mutex&    GetMTX();
        RIN auto      CreateLock()       { return GetMTX().CreateLock(); }
        RIN auto      CreateLock() const { return GetMTX().CreateLock(); }
        RIN void      Unlock()           { return GetMTX().Unlock(); }
        RIN void      Unlock()     const { return GetMTX().Unlock(); }

    protected:
                xint MnMutexID = 0;
        istatic xint SnMutexCount = 0;
        istatic std::unordered_map<xint, RA::Mutex> SmMutex;
        istatic std::mutex SoMutex;
    };
};
