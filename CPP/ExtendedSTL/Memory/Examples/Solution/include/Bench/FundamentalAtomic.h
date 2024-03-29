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
            Num(const xint MnNew) : MnVal(MnNew) {}
            Num(const Num& Other) { The = Other; }
            void operator=(const Num& Other) { MnVal = Other.MnVal; }
            RA::Atomic<xint> MnVal = 0;  // << Fundamental Atomic
        };
    private:
        RA::SharedPtr<Num> MoNumPtr;  // << Not Atomic
    public:
        FundamentalAtomic(){ MoNumPtr = RA::MakeShared<Num>(0); };

        FundamentalAtomic(const Num& FPint){ MoNumPtr = RA::MakeShared<Num>(FPint); };

        FundamentalAtomic(const FundamentalAtomic& Other) { *this = Other; }

        void operator=(const FundamentalAtomic& Other){ MoNumPtr = Other.MoNumPtr; }

        void Clear(){ MoNumPtr.Get().MnVal = 0; }

        void AddValue(const xint FPint)
        {
            MoNumPtr.Get().MnVal += FPint;
        }

        inline xint Get(){ return MoNumPtr.Get().MnVal.GetCopy(); }
    };
};