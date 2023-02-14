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
            Num(const uint FPint) : MnVal(FPint) {}
            Num(const Num& Other) { The = Other; }
            void operator=(const Num& Other) { MnVal = Other.MnVal; }
            uint MnVal = 0; // << Not Atomic
        };
    private:
        RA::SharedPtr<Num> MoNumPtr; // Not Atomic
    public:
        NoAtomic() { MoNumPtr = RA::MakeShared<Num>(0); };

        NoAtomic(const Num& FPint) { MoNumPtr = RA::MakeShared<Num>(FPint); };

        NoAtomic(const NoAtomic& Other) { *this = Other; }

        void operator=(const NoAtomic& Other) { MoNumPtr = Other.MoNumPtr; }

        void Clear() { MoNumPtr.Get().MnVal = 0; }

        void AddValue(const uint FPint)
        {
            if (Mtx.IsNull())
                Mtx = RA::SharedPtr<RA::Mutex>();
            auto Lock = Mtx.Get().CreateLock();
            MoNumPtr.Get().MnVal += FPint;
        }

        inline uint Get() { return MoNumPtr.Get().MnVal; }
    };
};