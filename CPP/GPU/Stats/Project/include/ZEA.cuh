#pragma once

#ifndef UsingMSVC
#include "ImportCUDA.cuh"
#endif

#include "Macros.h"

namespace RA
{
    // Zero Lag Exponential Average
    class ZEA
    {
        friend class Stats;
        friend class Deviation;
        friend class MeanAbsoluteDeviation;
        friend class StandardDeviation;
    public:
        ZEA(const double* FvValues = nullptr,
            const xint*  FnInsertIdxPtr = nullptr,
            const xint   FnStorageSize = 0);
        ~ZEA();

        DXF void SetPeriodSize(const xint FnPeriod = 0); // Lager = More Lag but Smoother
        IXF auto GetPeriodSize() const { return MnPeriod; }

        DXF void CopyStats(const ZEA& Other);

        DXF void SetDefaultValues(const double FnDefaualt);

        IXF auto GetZEA() const { return MnZMA; }
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

        DXF void Update(const double FnValue, const double FnValueBack);
        DXF void ResetRunningSize() { MnRunningSize = 0; }

    private:
        const double* MvValues;
        const xint* MnInsertIdxPtr;
        const xint  MnStorageSize = 0;
        xint        MnLogiclaSize = 0;
        xint        MnRunningSize = 0;

        double  MnAvg = 0;
        double  MnZMA = 0;
        double  MnSum = 0;
        double  MnCurrentValue = 0;
        double  MnLastValue = 0;

        xint    MnPeriod = 0;
        xint    MnLag = 0;
        double  MnAlpha = 0;

        static xint SnDefaultPeriod;
    };
};
