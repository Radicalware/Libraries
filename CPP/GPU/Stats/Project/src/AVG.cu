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
    MnAvg = Other.MnAvg;
}

DXF void RA::AVG::SetDefaultValues(const double FnDefaualt)
{
    MnAvg = FnDefaualt;
    MnRunningSize = 1;
    MnSum = MnAvg * MnStorageSize;
}

DXF xint RA::AVG::GetLastIDX() const
{
    cvar& LnInsert = *MnInsertIdxPtr;
    if (LnInsert == 0)
        return MnStorageSize - 1;
    return LnInsert - 1;
}

DXF void RA::AVG::Update(const double FnValue)
{
    if (MvValues)
    {
        MnRunningSize++;
        if (MnRunningSize >= MnStorageSize)
            MnRunningSize = MnStorageSize;

        const auto LnIdx = *MnInsertIdxPtr;
        auto LnOldestValue = 0.0;
        if (LnIdx >= (MnStorageSize - 1))
            LnOldestValue = MvValues[0];
        else
            LnOldestValue = MvValues[LnIdx + 1];


        MnNextSum += FnValue;
        MnAvg = MnNextSum / MnRunningSize;
        
        MnSum = MnNextSum;
        MnNextSum -= LnOldestValue;
    }
    else
    {
        MnAvg = ((MnAvg * MnLogiclaSize) + FnValue) / (MnLogiclaSize + 1);
        if (MnMaxTraceSize == 0 || MnMaxTraceSize > MnLogiclaSize)
            MnLogiclaSize++;
        MnSum += FnValue;
    }
}

