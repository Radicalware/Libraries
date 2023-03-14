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
    public:
        AVG(const double* FvValues,
            const xint    FnLogicalSize,
            const xint   *FnStorageSizePtr,
            const xint   *FnInsertIdxPtr);

        IXF auto GetAVG() const { return MnAvg; }
        IXF auto GetSum() const { return MnSum; }

        IXF auto BxUseStorageValues() const { return MbUseStorageValues; }
        IXF auto GetStorageSize()     const { return *MnStorageSizePtr; }
        IXF auto GetLogicalSize()     const { return MnLogicalSize; }
        IXF auto GetInsertIdx()       const { return *MnInsertIdxPtr; }
        IXF auto GetValues()          const { return MvValues; }

        DXF void CopyStats(const AVG& Other);

    private:
        DXF void Update();
        DXF void Update(const double FnValue);
        DXF void SetDefaultValues(const double FnDefaualt);

        const double* MvValues = nullptr;
              xint    MnLogicalSize = 0;
        const xint   *MnStorageSizePtr;
        const xint   *MnInsertIdxPtr;

        const bool    MbUseStorageValues;
              double  MnAvg = 0;
              double  MnSum = 0;
    };
};
