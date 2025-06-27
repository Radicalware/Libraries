#if UsingMSVC
#include "AVG.h"
#else
#include "AVG.cuh"
#endif

RA::AVG::AVG(
    const double* FvValues,
    const xint*   FnInsertIdxPtr,
    const xint    FnStorageSize)
    :
    MvValues(FvValues),
    MnInsertIdxPtr(FnInsertIdxPtr),
    MnStorageSize(FnStorageSize)
{
}

DXF void RA::AVG::CopyStats(const RA::AVG& Other)
{
    MnLogiclaSize = Other.MnLogiclaSize;
    MnRunningSize = Other.MnRunningSize;
    MnLastValue   = Other.GetLastValue();

    MnAvg = Other.MnAvg;
    MnSum = Other.MnSum;
    SetDebugErrorLevel(MnAvg);
}

DXF void RA::AVG::SetDefaultValues(const double FnDefaualt)
{
    MnRunningSize = MnStorageSize;
    MnAvg = FnDefaualt;
    MnSum = MnAvg * MnRunningSize;
    MnLogiclaSize = 0;
}

DXF xint RA::AVG::GetThisIDX() const
{
    return *MnInsertIdxPtr;
}

DXF xint RA::AVG::GetLastIDX() const
{
    cvar LnInsert = *MnInsertIdxPtr + 1;
    if (LnInsert >= MnStorageSize)
        return 0;
    return LnInsert;
}

DXF void RA::AVG::Update(const double FnValue)
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
        MnAvg  = MnSum / MnRunningSize;

        MnLastValue = (MnRunningSize < MnStorageSize) // last value not in MvValues
            ? 0
            : MvValues[(LnNextIdx >= MnStorageSize) ? 0 : LnNextIdx];
    }
    else
    {
        MnLastValue = MnCurrentValue;
        MnCurrentValue = FnValue;
        MnAvg = ((MnAvg * MnLogiclaSize) + FnValue) / (MnLogiclaSize + 1);
        MnSum += FnValue;
        MnLogiclaSize++;
    }
}

