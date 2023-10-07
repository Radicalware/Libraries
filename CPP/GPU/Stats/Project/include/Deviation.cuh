#pragma once

#include "xmap.h"

#ifdef UsingMSVC
#include "Deviation.h"
#include "Support.h"
#include "AVG.h"
#else
#include "Deviation.cuh"
#include "Support.cuh"
#include "ImportCUDA.cuh"
#include "AVG.cuh"
#endif

#include "Macros.h"

namespace RA
{
    class Deviation
    {
    public:
        enum class EType
        {
            None,
            MAD,
            SD
        };
    private:
          AVG* MoAvgPtr       = nullptr;
         EType MeType         = EType::None;

        double MnDeviation    = 0;
        double MnSumDeviation = 0; // For Logic Size 0

        double MnSumOffset    = 0;
        double MnLastOffDiff  = 0;
        double MnAvgOffset    = 0;

        double MnLastAvg      = 0;

        double MnCurrentVal   = 0;
        double MnLastVal      = 0;

        double MnMaxTraceSize = 0;

    public:
        Deviation(RA::AVG* FoAvgPtr, const EType FeType);

        DXF void   Update(const double FnValue);

        DXF void   CopyStats(const Deviation& Other);
        DXF void   SetDefaultValues();
        IXF void   SetAvg(AVG* FoAvgPtr) { MoAvgPtr = FoAvgPtr; }
        IXF void   SetMaxTraceSize(const xint FSize) { MnMaxTraceSize = FSize; }

        IXF auto   GetDeviation()   const { return MnDeviation; }
        IXF auto   GetAvgOffset()   const { return MnAvgOffset; }
        IXF auto   GetLastOffDiff() const { return std::abs(MnLastOffDiff); }
        IXF auto   GetLastDirectionalOffset() const { return MnLastOffDiff; }

        DXF double GetDifference(const double FnCurrent, const double FnLast) const;
        DXF double GetDifference(const double FnCurrent) const;
        DXF double GetDifference() const;

        DXF double GetDirectionalOffset(const double FnCurrent, const double FnLast) const;
        DXF double GetDirectionalOffset(const double FnCurrent) const;
        DXF double GetDirectionalOffset() const;

        DXF double GetOffset(const double FnCurrent, const double FnLast) const;
        DXF double GetOffset(const double FnCurrent) const;
        DXF double GetOffset() const;

        DXF double GetFractinalOffset(const double FnCurrent, const double FnLast) const;
        DXF double GetFractinalOffset(const double FnCurrent) const;
        DXF double GetFractinalOffset() const;

        IXF auto   GetStorageSize() const { return (*MoAvgPtr).GetStorageSize(); }
        IXF auto   GetLogicalSize() const { return (*MoAvgPtr).GetLogicalSize(); }
        IXF auto   GetInsertIdx()   const { return (*MoAvgPtr).GetInsertIdx(); }

        // DXF virtual void   Update(const double FnValue) = 0;

        IXF void   SetCurrentValue(const double FnValue) { MnCurrentVal = FnValue; }
        IXF double GetCurrentValue() const { return MnCurrentVal; }
        IXF double GetLastValue() const { return MnLastVal; }
        IXF auto   BxIncreasing() const { return !(MnLastVal > MnCurrentVal); }

        // note: Default FnLast = 0 not used b/c you don't want a case where you give it 0 and then get the "Last" value which isn't zero

        DXF static double GetFractional(double FnValue);
    };
};
