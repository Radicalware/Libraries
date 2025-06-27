#pragma once

#ifndef UsingMSVC
#include "ImportCUDA.cuh"
#else
#include "xvector.h"
#endif
#include "RawMapping.h"


namespace RA
{
    class Omaha
    {
        friend class Stats;
    public:
        Omaha(
            const EHardware FeHardware,
            const double* FvValues,
            const   xint* FnInsertIdxPtr,
            const   xdbl* FnMinPtr,
            const   xdbl* FnMaxPtr,
            const   xint  FnStorageSize = 0);

        ~Omaha();

        DXF void CopyStats(const Omaha& Other);
        DXF void Update();
        DXF void Update(const double FnValue);
    private:
        DXF void SetDefaultValues(const double FnValue);
        DXF void Insert(const double FnValue);
        DXF void Remove(double FnValue);
        DXF void RemoveOldest();
    public:

        IXF auto GetMax()         const { return *MnMaxPtr; }
        IXF auto GetMin()         const { return *MnMinPtr; }
        IXF auto GetRunningSize() const { return MnRunningSize; }

        IXF auto BxFullSize()     const { return MnStorageSize == MnRunningSize; }

        DXF xint OldIndexFor(const double FnValue) const;
        DXF xint NewIndexFor(const double Value) const;

        IXF xint GetMaxIdx() const { return NewIndexFor(GetMax()); }
        IXF xint GetMinIdx() const { return NewIndexFor(GetMin()); }

        DXF bool BxHigh() const;
        DXF bool BxLow()  const;

        DXF double GetHighIdxScaled() const;
        DXF double GetLowIdxScaled() const;

        IXF double GetLocalMin() const { return *MvSet.cbegin();  }
        IXF double GetlocalMax() const { return *MvSet.crbegin(); }

    private:
        EHardware MeHardware = EHardware::Default;
        const double* MvValues = nullptr; // end point slides with MnInsertIdxPtr
        const xint* MnInsertIdxPtr;
        xint        MnStorageSize;
        const xdbl* MnMinPtr = nullptr;
        const xdbl* MnMaxPtr = nullptr;
        xint        MnRunningSize = 0;

        std::list<double> MvTimeseries;
        std::unordered_multimap<double, std::list<double>::iterator> MmDblToIndex;
        std::multiset<double> MvSet;
    };
}