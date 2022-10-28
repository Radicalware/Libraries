#pragma once


#include <unordered_map>

#include "Atomic.h"
#include "SharedPtr.h"
#include "Mutex.h"


namespace RA
{
    class MutexHandler
    {
    public:
                    MutexHandler();
        bool        UsingMutex() const;
        RA::Mutex&  GetMutex() const;

    protected:
        size_t MnMutexID = 0;
        inline static size_t SnMutexCount = 0;
        inline static std::unordered_map<size_t, RA::Mutex> SmMutex;
    };
};