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

        STOCH(const double* FvValues = nullptr,
              const   xint* FnInsertIdxPtr = nullptr,
              const   xint  FnStorageSize = 0);

        IXF auto GetMax()       const { return MnBiggest; }
        IXF auto GetCurrent()   const { return (MvValues) ? MvValues[*MnInsertIdxPtr] : MnLast; }
        IXF auto GetMin()       const { return MnSmallest; }
        IXF auto GetSTOCH()     const { return MnSTOCH; }

        IXF auto BxNoEntry() const { return MnSmallest == DBL_MAX; }

        DXF void CopyStats(const STOCH& Other);

    private:
        DXF void Update();
        DXF void Update(const double FnValue);
        DXF void SetDefaultValues(const double FnDefaualt);

        const double* MvValues;
        const xint*   MnInsertIdxPtr;
        const xint    MnStorageSize;

        double  MnBiggest  = -DBL_MAX;
        double  MnSmallest =  DBL_MAX;
        double  MnLast = 0;
        double  MnSTOCH = 0;
        xint    MnRunningSize = 0;
    };
};
