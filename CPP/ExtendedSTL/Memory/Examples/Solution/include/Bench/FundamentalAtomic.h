#pragma once

#include "Bench/BenchBase.h"
#include "Atomic.h"

namespace Bench
{
    class FundamentalAtomic : public Base
    {
    public:
        struct Num
        {
            Num() {}
            Num(const pint FPint) : MnVal(FPint) {}
            Num(const Num& Other) { The = Other; }
            void operator=(const Num& Other) { MnVal = Other.MnVal; }
            RA::Atomic<pint> MnVal = 0;  // << Fundamental Atomic
        };
    private:
        RA::SharedPtr<Num> MoNumPtr;  // << Not Atomic
    public:
        FundamentalAtomic(){ MoNumPtr = MakePtr<Num>(0); };

        FundamentalAtomic(const Num& FPint){ MoNumPtr = MakePtr<Num>(FPint); };

        FundamentalAtomic(const FundamentalAtomic& Other) { *this = Other; }

        void operator=(const FundamentalAtomic& Other){ MoNumPtr = Other.MoNumPtr; }

        void Clear(){ MoNumPtr.Get().MnVal = 0; }

        void AddValue(const pint FPint)
        {
            MoNumPtr.Get().MnVal += FPint;
        }

        inline pint Get(){ return MoNumPtr.Get().MnVal.GetCopy(); }
    };
};