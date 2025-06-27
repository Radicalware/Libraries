#if UsingMSVC
#include "ZEA.h"
#else
#include "ZEA.cuh"
#endif

xint RA::ZEA::SnDefaultPeriod = 20;

RA::ZEA::ZEA(
    const double* FvValues,
    const xint*   FnInsertIdxPtr,
    const xint    FnStorageSize)
    :
    MvValues(FvValues),
    MnInsertIdxPtr(FnInsertIdxPtr),
    MnStorageSize(FnStorageSize)
{
    cvar LnPeriod = (MnStorageSize > SnDefaultPeriod) ? SnDefaultPeriod : MnStorageSize;
    SetPeriodSize(LnPeriod);
}

RA::ZEA::~ZEA()
{
}

DXF void RA::ZEA::SetPeriodSize(const xint FnPeriod)
{
    if (FnPeriod == 0)
        MnPeriod = MnStorageSize;
    else
        MnPeriod = FnPeriod;
    MnLag = (xint)(((double)FnPeriod - 1.0) / 2.0);
    MnAlpha = (2.0 / (FnPeriod + 1.0));
    MnZMA = (0.0);
}

DXF void RA::ZEA::CopyStats(const RA::ZEA& Other)
{
    The.SetDefaultValues(Other.MnZMA);

    MnLogiclaSize = Other.MnLogiclaSize;

    MnCurrentValue = Other.MnCurrentValue;
    MnLastValue = Other.GetLastValue();

    MnAvg = Other.MnAvg;
    MnZMA = Other.MnZMA;
    MnSum = Other.MnSum;

    MnPeriod = Other.MnPeriod;
    MnLag = Other.MnLag;
    MnAlpha = Other.MnAlpha;
}

DXF void RA::ZEA::SetDefaultValues(const double FnDefaualt)
{
    MnRunningSize = MnStorageSize;
    MnZMA = FnDefaualt;
    MnAvg = FnDefaualt;
    MnSum = MnAvg * MnRunningSize;
    MnLogiclaSize = 0;
    SetPeriodSize(SnDefaultPeriod);
}

DXF xint RA::ZEA::GetThisIDX() const
{
    return *MnInsertIdxPtr;
}

DXF xint RA::ZEA::GetLastIDX() const
{
    cvar LnInsert = *MnInsertIdxPtr + 1;
    if (LnInsert >= MnStorageSize)
        return 0;
    return LnInsert;
}

DXF void RA::ZEA::Update(const double FnValue, const double FnValueBack)
{
    if (MvValues)
    {
        MnCurrentValue = FnValue;
        MnRunningSize++;
        if (MnRunningSize >= MnStorageSize)
            MnRunningSize = MnStorageSize;

        const auto LnIdx = *MnInsertIdxPtr;
        const auto LnNextIdx = LnIdx + 1;

        MnSum -= MnLastValue;
        MnSum += MvValues[LnIdx];
        MnAvg = MnSum / MnRunningSize;

        MnLastValue = (MnRunningSize < MnStorageSize) // last value not in MvValues
            ? 0
            : MvValues[(LnNextIdx >= MnStorageSize) ? 0 : LnNextIdx];

        cvar LnVal = MvValues[LnIdx];

        auto LnPriceLag = FnValue + (FnValue - FnValueBack);


        if (MnRunningSize >= MnPeriod) {
            MnZMA = MnAlpha * LnPriceLag + (1.0 - MnAlpha) * MnZMA;
        }
        else {
            MnZMA = FnValue;
        }
    }
    else
    {
        ThrowIt("Storage Size Required");
    }
}

