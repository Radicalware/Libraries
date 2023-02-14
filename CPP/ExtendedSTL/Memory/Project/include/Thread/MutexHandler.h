#pragma once

#include <unordered_map>

#include "SharedPtr.h"
#include "Mutex.h"

namespace RA
{
    class MutexHandler
    {
    public:
        INL constexpr MutexHandler() noexcept {}
        void          CreateMTX(); // Warning: there is a windows macro "CreateMutex"
        bool          UsingMTX() const;
        RA::Mutex&    GetMTX() const;
        RA::Mutex&    GetMTX();
        INL RA::Lock  CreateLock()       { return GetMTX().CreateLock(); }
        INL RA::Lock  CreateLock() const { return GetMTX().CreateLock(); }
        INL void      Unlock()           { return GetMTX().Unlock(); }
        INL void      Unlock()     const { return GetMTX().Unlock(); }

    protected:
        size_t MnMutexID = 0;
        inline static size_t SnMutexCount = 0;
        inline static std::unordered_map<size_t, RA::Mutex> SmMutex;
    };
};
