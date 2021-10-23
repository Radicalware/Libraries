#pragma once

// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Atomic/BaseAtomic.h"

namespace RA
{
#ifndef __ATOMIC_DEFINED__
#define __ATOMIC_DEFINED__
    template<typename T, typename enabler_t = void> class Atomic;
#endif

    template<typename T>
    class Atomic<T, typename std::enable_if_t<std::is_fundamental<T>::value>> : public BaseAtomic<T>
    {
    public:
                                 Atomic() { The.store(static_cast<T>(0)); };
        template<class T>        Atomic(const T& Other) { The.store(Other); }

        template<class O>
        typename std::enable_if<!std::is_same<O, Atomic<O>>::value, void>::type  
            operator=(const O& Other)  { The.store(static_cast<T>(Other)); }

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
        T GetCopy() const { return The.load(); }
        std::atomic<T>& Base() const { return *reinterpret_cast<std::atomic<T>>(this); }
    };
};
