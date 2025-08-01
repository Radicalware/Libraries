﻿#pragma once

#ifndef UsingMSVC
#include "ImportCUDA.cuh"
#endif

#include "Macros.h"

namespace RA
{
    class AVG
    {
        friend class Stats;
        friend class Deviation;
        friend class MeanAbsoluteDeviation;
        friend class StandardDeviation;
    public:
        AVG(const double* FvValues = nullptr,
            const xint* FnInsertIdxPtr = nullptr,
            const xint    FnStorageSize = 0);

        DXF void CopyStats(const AVG& Other);

        DXF void SetDefaultValues(const double FnDefaualt);

        IXF auto GetAVG() const { return MnAvg; }
        IXF auto GetSum() const { return MnSum; }

        IXF auto BxUseStorageValues() const { return  MvValues != nullptr; }
        IXF auto GetStorageSize()     const { return  MnStorageSize; }
        IXF auto GetLogicalSize()     const { return  MnLogiclaSize; }
        IXF auto GetRunningSize()     const { return  MnRunningSize; }
        IXF auto GetInsertIdx()       const { return *MnInsertIdxPtr; }
        IXF auto GetValues()          const { return  MvValues; }

        IXF double GetCurrenetValue() const { return MnCurrentValue; }
        IXF double GetThisValue()     const { return MvValues[GetThisIDX()]; }
        IXF double GetLastValue()     const { return MvValues[GetLastIDX()]; }
        DXF xint   GetThisIDX()       const;
        DXF xint   GetLastIDX()       const;

        DXF void Update(const double FnValue);
        DXF void ResetRunningSize() { MnRunningSize = 0; }

    private:
        const double* MvValues;
        const xint* MnInsertIdxPtr;
        const xint  MnStorageSize;
        xint        MnLogiclaSize = 0;
        xint        MnRunningSize = 0;

        double  MnAvg = 0;
        double  MnSum = 0;
        double  MnCurrentValue = 0;
        double  MnLastValue = 0;
    };
};
