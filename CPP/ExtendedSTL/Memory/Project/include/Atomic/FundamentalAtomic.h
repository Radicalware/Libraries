#pragma once

// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Mutex.h"
#include "Thread/MutexHandler.h"

namespace RA
{
#ifndef __ATOMIC_DEFINED__
#define __ATOMIC_DEFINED__
    template<typename T, typename enabler_t = void>
    class Atomic;

    template<typename T> class Atomic<T, typename std::enable_if_t< BxFundamental(T) || BxSimple(T)>>;
    template<typename T> class Atomic<T, typename std::enable_if_t<!BxFundamental(T) && !BxSimple(T)>>;
#endif
};

namespace RA
{
    template<typename T>
    class Atomic<T, typename std::enable_if_t<BxFundamental(T) || BxSimple(T)>> : public std::atomic<T>, public MutexHandler
    {
    public:
        using std::atomic<T>::atomic;

        template <typename O> CIN typename std::enable_if<!BxSameType(T, O), void>::type
        /*void*/ operator=(const O& Other) { The.store(static_cast<T>(Other)); }

        template <typename O> CIN typename std::enable_if< BxSameType(T, O), void>::type
        /*void*/ operator=(const O& Other) { The.store(Other); }

                            CIN void operator=(const Atomic<T>& Other)  { The.store(Other.GetCopy()); }
                            //void operator++(int)            { The.store(The.load() + 1); }
                            //void operator++()               { The.store(The.load() + 1); }
                            //void operator--(int)            { The.store(The.load() - 1); }
                            //void operator--()               { The.store(The.load() - 1); }
        template<class O>   CIN void operator*=(const O Other)  { The.store(The.load() * static_cast<T>(Other)); }
        template<class O>   CIN void operator+=(const O Other)  { The.store(The.load() + static_cast<T>(Other)); }
        template<class O>   CIN void operator-=(const O Other)  { The.store(The.load() - static_cast<T>(Other)); }
        template<class O>   CIN void operator/=(const O Other)  { The.store(The.load() / static_cast<T>(Other)); }
        template<class O>   CIN T    operator% (const O Other)  { return The.load() % static_cast<T>(Other); }
        template<class O>   CIN void operator%=(const O Other)  { The.store(The.load() % static_cast<T>(Other)); }
        // + Other && - Other is included in base;

        CIN bool operator!() { return !The.load(); }
        CIN T GetCopy() const { return The.load(); }
    };
};
