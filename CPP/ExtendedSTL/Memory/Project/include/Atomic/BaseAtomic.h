#pragma once

// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include <memory>

#include "Thread/Mutex.h"

namespace RA
{
    template<typename T> 
    class BaseAtomic : public std::atomic<T>
    {
    protected:
        RA::SharedPtr<RA::Mutex> Mtx;

    public:
        constexpr BaseAtomic() noexcept {};
        constexpr BaseAtomic(nullptr_t) noexcept {}

        using  std::atomic<T>::atomic;
        inline RA::Mutex&               GetMutex() { 
            if (Mtx == nullptr) Mtx = RA::MakeShared<RA::Mutex>(); return Mtx.Get(); 
        }
        inline RA::SharedPtr<RA::Mutex> GetMutexPtr() { 
            if (Mtx == nullptr) Mtx = RA::MakeShared<RA::Mutex>(); return Mtx; 
        }
    };
};