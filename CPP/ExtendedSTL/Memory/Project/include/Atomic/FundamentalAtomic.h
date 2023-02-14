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

    template<typename T> class Atomic<T, typename std::enable_if_t< IsFundamental(T) || IsSimple(T)>>;
    template<typename T> class Atomic<T, typename std::enable_if_t<!IsFundamental(T) && !IsSimple(T)>>;
#endif
};

namespace RA
{
    template<typename T>
    class Atomic<T, typename std::enable_if_t<IsFundamental(T) || IsSimple(T)>> : public std::atomic<T>, public MutexHandler
    {
    public:
        using std::atomic<T>::atomic;

        template <typename O> inline typename std::enable_if<!IsSame(T, O), void>::type
        /*void*/ operator=(const O& Other) { The.store(static_cast<T>(Other)); }

        template <typename O> inline typename std::enable_if< IsSame(T, O), void>::type
        /*void*/ operator=(const O& Other) { The.store(Other); }

                            void operator=(const Atomic<T>& Other)  { The.store(Other.GetCopy()); }
                            void operator++(int)            { The.store(The.load() + 1); }
                            void operator++()               { The.store(The.load() + 1); }
                            void operator--(int)            { The.store(The.load() - 1); }
                            void operator--()               { The.store(The.load() - 1); }
        template<class O>   void operator*=(const O Other)  { The.store(The.load() * static_cast<T>(Other)); }
        template<class O>   void operator+=(const O Other)  { The.store(The.load() + static_cast<T>(Other)); }
        template<class O>   void operator-=(const O Other)  { The.store(The.load() - static_cast<T>(Other)); }
        template<class O>   void operator/=(const O Other)  { The.store(The.load() / static_cast<T>(Other)); }
        template<class O>   T    operator% (const O Other)  { return The.load() % static_cast<T>(Other); }
        template<class O>   void operator%=(const O Other)  { The.store(The.load() % static_cast<T>(Other)); }
        // + Other && - Other is included in base;

        bool operator!() { return !The.load(); }
        T GetCopy() const { return The.load(); }
    };
};
