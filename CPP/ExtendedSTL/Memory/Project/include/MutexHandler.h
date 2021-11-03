#pragma once


#include "SharedPtr.h"
#include "Mutex.h"


namespace RA
{
    class MutexHandler
    {
    protected:
        RA::SharedPtr<RA::Mutex> MoMutexPtr;
    public:
              void                       InitMutex();
              RA::Mutex&                 GetMutex();
        const RA::Mutex&                 GetMutex() const;
              RA::SharedPtr<RA::Mutex>&  GetMutexPtr();
        const RA::SharedPtr<RA::Mutex>&  GetMutexPtr() const;
    };
};