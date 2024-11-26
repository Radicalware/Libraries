#pragma once

// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "RawMapping.h"

namespace RA
{
#ifndef __ATOMIC_DEFINED__
#define __ATOMIC_DEFINED__
    template<typename T, typename enabler_t = void>
                         class Atomic;
                         
    template<typename T> class Atomic<T, typename std::enable_if_t< BxFundamental(T) ||  BxSimple(T)>>;
    template<typename T> class Atomic<T, typename std::enable_if_t<!BxFundamental(T) && !BxSimple(T)>>;
#endif
};

#include "Atomic/ClassAtomic.h"
#include "Atomic/FundamentalAtomic.h"

