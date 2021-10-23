#pragma once

#include "Mutex.h"
#include "SharedPtr.h"
#include "Atomic.h"


namespace Bench
{
    class Base
    {
    public:
        RA::SharedPtr<RA::Mutex> Mtx = MakePtr<RA::Mutex>();
        inline static const pint MnLooper = 200000;
    };
};