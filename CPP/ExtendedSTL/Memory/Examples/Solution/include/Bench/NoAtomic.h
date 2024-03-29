#pragma once

#include "Bench/BenchBase.h"


namespace Bench
{
    class NoAtomic : public Base
    {
    public:
        struct Num
        {
            Num() {}
            Num(const xint FPint) : MnVal(FPint) {}
            Num(const Num& Other) { The = Other; }
            void operator=(const Num& Other) { MnVal = Other.MnVal; }
            xint MnVal = 0; // << Not Atomic
        };
    private:
        RA::SharedPtr<Num> MoNumPtr; // Not Atomic
    public:
        CIN NoAtomic() { MoNumPtr = RA::MakeShared<Num>(0); };
        CIN NoAtomic(const Num& FPint) { MoNumPtr = RA::MakeShared<Num>(FPint); };
        CIN NoAtomic(const NoAtomic& Other) { *this = Other; }

        CIN void operator=(const NoAtomic& Other) { MoNumPtr = Other.MoNumPtr; }

        RIN void Clear() { MoNumPtr.Get().MnVal = 0; }

        RIN void AddValue(const xint FPint)
        {
            if (Mtx.IsNull())
                Mtx = RA::SharedPtr<RA::Mutex>();
            auto Lock = Mtx.Get().CreateLock();
            MoNumPtr.Get().MnVal += FPint;
        }

        RIN xint Get() { return MoNumPtr.Get().MnVal; }
    };
};