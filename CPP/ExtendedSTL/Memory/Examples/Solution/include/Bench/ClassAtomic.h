#pragma once

#include "Bench/BenchBase.h"

namespace Bench
{
    class ClassAtomic : public Base
    {
    public:
        struct Num
        {
            xint MnVal = 0; // Not Atomic
            DefaultConstruct(Num);

#if _HAS_CXX20
            Num(const xint FPint) : MnVal(FPint) {}
#endif

        };
    private:
        RA::Atomic<Num> MoNumAtm; // Class Atomic
    public:
        ClassAtomic() 
        { 
            MoNumAtm = Num();
        };
         
        ClassAtomic(const Num& FnNew){ MoNumAtm = Num{ FnNew }; };

        ClassAtomic(const ClassAtomic& Other) { *this = Other; }

        void operator=(const ClassAtomic& Other){ MoNumAtm = Other.MoNumAtm; }

        void Clear() { MoNumAtm = Num(); }

        void AddValue(const xint Fuint)
        {
            auto Lock = Mtx.Get().CreateLock();
            MoNumAtm = Num{ MoNumAtm.GetCopy().MnVal + Fuint };
        }

        inline xint Get(){ return MoNumAtm.GetCopy().MnVal; }
    };
};