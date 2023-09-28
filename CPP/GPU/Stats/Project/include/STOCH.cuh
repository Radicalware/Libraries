#pragma once
// Copyright via Apache v2 Licence [2023][Joel Leagues aka Scourge]

#ifndef UsingMSVC
#include "ImportCUDA.cuh"
#endif

#include "Macros.h"
#include <limits>

namespace RA
{
    class STOCH
    {
    public:
        friend class Stats;

        STOCH(
            const double* FvValues,
            const xint    FnLogicalSize,
            const xint   *FnStorageSizePtr,
            const xint   *FnInsertIdxPtr);

        IXF auto GetMax()       const { return MnBiggest; }
        IXF auto GetCurrent()   const { return MvValues[(*MnInsertIdxPtr > 0) ? *MnInsertIdxPtr : *MnStorageSizePtr - 1]; }
        IXF auto GetMin()       const { return MnSmallest; }
        IXF auto GetSTOCH()     const { return MnSTOCH; }

        IXF auto BxNoEntry() const { return MnSmallest == DBL_MAX; }

        DXF void CopyStats(const STOCH& Other);

    private:
        DXF void Update();
        DXF void Update(const double FnValue);
        DXF void SetLogicalSize(const xint FnLogicalSize);
        DXF void SetDefaultValues(const double FnDefaualt);

        const double* MvValues;
              xint    MnLogicalSize = 0;
        const xint   *MnStorageSizePtr;
        const xint   *MnInsertIdxPtr;

        double  MnBiggest  = -DBL_MAX;
        double  MnSmallest =  DBL_MAX;
        double  MnSTOCH = 0;
    };
};
