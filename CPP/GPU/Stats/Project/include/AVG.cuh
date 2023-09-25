#pragma once

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
        AVG(const double* FvValues,
            const xint    FnLogicalSize,
            const xint* FnStorageSizePtr,
            const xint* FnInsertIdxPtr);

        IXF void SetMaxTraceSize(const xint FSize) { MnMaxTraceSize = FSize; }

        DXF void CopyStats(const AVG& Other);

        DXF void SetDefaultValues(const double FnDefaualt);

        IXF auto GetAVG() const { return MnAvg; }
        IXF auto GetSum() const { return MnSum; }

        IXF auto BxUseStorageValues() const { return  MbUseStorageValues; }
        IXF auto GetStorageSize()     const { return *MnStorageSizePtr; }
        IXF auto GetLogicalSize()     const { return  MnLogicalSize; }
        IXF auto GetRunningSize()     const { return  MnRunningSize; }
        IXF auto GetInsertIdx()       const { return *MnInsertIdxPtr; }
        IXF auto GetValues()          const { return  MvValues; }

        IXF double GetOldValue() const { return MvValues[GetOldIDX()]; }
        DXF xint   GetOldIDX() const;

        DXF void Update(const double FnValue);
        DXF void ResetRunningSize() { MnRunningSize = 0; }

    private:
        const double* MvValues = nullptr;
        xint    MnLogicalSize = 0;
        const xint* MnStorageSizePtr;
        const xint* MnInsertIdxPtr;

        xint    MnRunningSize = 0;
        xint    MnMaxTraceSize = 0;
        const bool    MbUseStorageValues;
        double  MnAvg = 0;
        double  MnSum = 0;
    };
};
