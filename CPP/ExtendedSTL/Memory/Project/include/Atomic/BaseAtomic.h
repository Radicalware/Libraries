#pragma once

// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include <memory>

#include "Mutex.h"
#include "SharedPtr.h"

namespace RA
{
    template<typename T> 
    class BaseAtomic : public std::atomic<T>
    {
    protected:
        RA::SharedPtr<RA::Mutex> Mtx;
    public:
        using  std::atomic<T>::atomic;
        inline RA::Mutex&               GetMutex()    { if (!Mtx) Mtx = MakePtr<RA::Mutex>(); return Mtx.Get(); }
        inline RA::SharedPtr<RA::Mutex> GetMutexPtr() { if (!Mtx) Mtx = MakePtr<RA::Mutex>(); return Mtx; }

    };
};