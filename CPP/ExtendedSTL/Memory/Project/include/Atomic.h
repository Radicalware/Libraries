#pragma once

// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

namespace RA
{
#ifndef __ATOMIC_DEFINED__
#define __ATOMIC_DEFINED__
    template<typename T, typename enabler_t = void> class Atomic;
#endif
};

#include "Atomic/BaseAtomic.h"
#include "Atomic/ClassAtomic.h"
#include "Atomic/FundamentalAtomic.h"

