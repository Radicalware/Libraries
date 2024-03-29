#pragma once

#include "Mutex.h"
#include "SharedPtr.h"
#include "Atomic.h"


namespace Bench
{
    class Base
    {
    public:
        RA::SharedPtr<RA::Mutex> Mtx = RA::MakeShared<RA::Mutex>();
        inline static const xint MnLooper = 200;
    };
};