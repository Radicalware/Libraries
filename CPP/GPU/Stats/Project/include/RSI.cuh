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
            const uint    FnLogicalSize,
            const uint   *FnStorageSizePtr,
            const uint   *FnInsertIdxPtr);

        DXF double GetCurvedRSI() const;
        IXF auto   GetRSI() const { return MnRSI; }
        IXF auto   GetLogicalSize() const { return MnLogicalSize; }
        DXF void   CopyStats(const RSI& Other);

    private:
        DXF void Update();
        DXF void SetDefaultValues(const double FnDefaualt = 50);

        const double* MvValues = nullptr;
        uint          MnLogicalSize = 0;
        const uint   *MnStorageSizePtr;
        const uint   *MnInsertIdxPtr;

        double  MnRSI = 0;
    };
};
