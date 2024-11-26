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

        STOCH(const double* FvValues,
              const   xint* FnInsertIdxPtr,
              const   xdbl* FnMinPtr,
              const   xdbl* FnMaxPtr,
              const   xint  FnStorageSize = 0);

        IXF auto GetMin()         const { return *MnMinPtr; }
        IXF auto GetMax()         const { return *MnMaxPtr; }
        IXF auto GetCurrent()     const { return (MvValues) ? MvValues[*MnInsertIdxPtr] : MnLast; }
        IXF auto GetSTOCH()       const { return MnSTOCH; }
        IXF auto GetScaledSTOCH() const { return (2.0 * (MnSTOCH / 100.0)) - 1.0; } // scalled between 1 to -1

        DXF void CopyStats(const STOCH& Other);

    private:
        DXF void Update();
        DXF void Update(const double FnValue);
        DXF void SetDefaultValues(const double FnDefaualt);

        const double* MvValues;
        const xint*   MnInsertIdxPtr;
        const xint    MnStorageSize;

        const xdbl* MnMinPtr = nullptr;
        const xdbl* MnMaxPtr = nullptr;

        double  MnLast = 0;
        double  MnSTOCH = 0;
        xint    MnRunningSize = 0;
    };
};
