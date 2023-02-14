#pragma once
// Copyright via Apache v2 Licence [2023][Joel Leagues aka Scourge]

#ifndef UsingMSVC
#include "ImportCUDA.cuh"
#endif

#include "Macros.h"

namespace RA
{
    class STOCH
    {
    public:
        friend class Stats;

        STOCH(
            const double* FvValues,
            const uint    FnLogicalSize,
            const uint   *FnStorageSizePtr,
            const uint   *FnInsertIdxPtr);

        IXF auto GetMax()       const { return MnBiggest; }
        IXF auto GetCurrent()   const { return MvValues[(*MnInsertIdxPtr > 0) ? *MnInsertIdxPtr - 1 : *MnStorageSizePtr - 1]; }
        IXF auto GetMin()       const { return MnSmallest; }
        IXF auto GetSTOCH()     const { return MnSTOCH; }

        IXF auto BxNoEntry() const { return MnSmallest == DBL_MAX; }

        DXF void CopyStats(const STOCH& Other);

    private:
        DXF void Update();
        DXF void SetLogicalSize(const uint FnLogicalSize);
        DXF void SetDefaultValues(const double FnDefaualt);

        const double* MvValues;
              uint    MnLogicalSize = 0;
        const uint   *MnStorageSizePtr;
        const uint   *MnInsertIdxPtr;

        double  MnBiggest = 0;
        double  MnSmallest = 0;
        double  MnSTOCH = 0;
    };
};
