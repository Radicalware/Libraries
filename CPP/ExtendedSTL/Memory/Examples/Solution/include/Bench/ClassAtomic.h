#pragma once

#include "Bench/BenchBase.h"

namespace Bench
{
    class ClassAtomic : public Base
    {
    public:
        struct Num
        {
            Num() {}
            Num(const pint FPint) : MnVal(FPint) {}
            Num(const Num& Other) { The = Other; }
            void operator=(const Num& Other) { MnVal = Other.MnVal; }
            pint MnVal = 0; // Not Atomic
        };
    private:
        RA::Atomic<Num>    MoNumAtm; // Class Atomic
    public:
        ClassAtomic(){ MoNumAtm = Num(0); };
         
        ClassAtomic(const Num& FPint){ MoNumAtm = MakePtr<Num>(FPint); };

        ClassAtomic(const ClassAtomic& Other) { *this = Other; }

        void operator=(const ClassAtomic& Other){ MoNumAtm = Other.MoNumAtm.GetCopy(); }

        void Clear() { MoNumAtm = Num(0); }

        void AddValue(const pint FPint)
        {
            auto Lock = Mtx.Get().CreateLock();
            RA::Atomic<Num> Val = Num(MoNumAtm.Get().MnVal + FPint);
            MoNumAtm = Val;
        }

        inline pint Get(){ return MoNumAtm.GetCopy().MnVal; }
    };
};