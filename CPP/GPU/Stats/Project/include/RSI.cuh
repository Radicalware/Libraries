﻿#pragma once

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
        RSI(const double* FvValues = nullptr,
            const   xint* FnInsertIdxPtr = nullptr,
            const   xint  FnStorageSize = 0);

        DXF double GetCurvedRSI() const;
        IXF auto   GetRSI() const { return MnRSI; }
        DXF void   CopyStats(const RSI& Other);

    private:
        DXF void Update();
        DXF void SetDefaultValues(const double FnDefaualt = 50);

        const double* MvValues;
        const xint*   MnInsertIdxPtr;
        const xint    MnStorageSize;

        double  MnRSI = 0;
        xint    MnRunningSize = 0;
    };
};
