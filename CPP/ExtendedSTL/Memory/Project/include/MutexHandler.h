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
        ~MutexHandler();
        void        CheckMutex();
        RA::Mutex&  GetMutex();
        RA::Mutex*  GetMutexPtr();

    protected:
        RA::Mutex* MoMutexPtr = nullptr;
    };
};