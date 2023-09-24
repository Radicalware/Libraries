#pragma once

#ifndef UsingMSVC
#include "ImportCUDA.cuh"
#endif

#include "Macros.h"

namespace RA
{
    class RSI
    {
        friend class Stats;
    public:
        RSI(const double* FvValues,
            const xint    FnLogicalSize,
            const xint   *FnStorageSizePtr,
            const xint   *FnInsertIdxPtr);

        DXF double GetCurvedRSI() const;
        IXF auto   GetRSI() const { return MnRSI; }
        IXF auto   GetLogicalSize() const { return MnLogicalSize; }
        DXF void   CopyStats(const RSI& Other);

    private:
        DXF void Update();
        DXF void SetDefaultValues(const double FnDefaualt = 50);

        const double* MvValues = nullptr;
        xint          MnLogicalSize = 0;
        const xint   *MnStorageSizePtr;
        const xint   *MnInsertIdxPtr;

        double  MnRSI = 0;
    };
};
