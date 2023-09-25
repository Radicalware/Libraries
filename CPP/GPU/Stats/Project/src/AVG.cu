#if UsingMSVC
#include "AVG.h"
#else
#include "AVG.cuh"
#endif

RA::AVG::AVG(
    const double* FvValues,
    const xint    FnLogicalSize,
    const xint* FnStorageSizePtr,
    const xint* FnInsertIdxPtr)
    :
    MvValues(FvValues),
    MnLogicalSize(FnLogicalSize),
    MnStorageSizePtr(FnStorageSizePtr),
    MnInsertIdxPtr(FnInsertIdxPtr),
    MbUseStorageValues(*FnStorageSizePtr > 0 && FnLogicalSize > 0)
{
    Begin();
    if (*MnStorageSizePtr == 0 && MnLogicalSize > 0)
        ThrowIt("If storage size is 0, then logical size must also be 0 to start");

    Rescue()
}

DXF void RA::AVG::CopyStats(const RA::AVG& Other)
{
    MnAvg = Other.MnAvg;
    MnLogicalSize = Other.MnLogicalSize;
}

DXF void RA::AVG::SetDefaultValues(const double FnDefaualt)
{
    MnAvg = FnDefaualt;
    MnRunningSize = 1;
    MnSum = MnAvg * MnLogicalSize;
}

DXF xint RA::AVG::GetOldIDX() const
{
    const auto& LnStorage = *The.MnStorageSizePtr;
    const auto  LnLogicSize = The.MnLogicalSize;
    const auto& LnStart = *The.MnInsertIdxPtr;

    return ((LnStart >= LnLogicSize)
        ? LnStart - LnLogicSize
        : (LnStorage - (LnLogicSize - LnStart)));
}

DXF void RA::AVG::Update(const double FnValue)
{
    if (MbUseStorageValues)
    {
        MnRunningSize++;
        if (MnRunningSize > MnLogicalSize)
            MnRunningSize = MnLogicalSize;
#if UsingMSVC
        if (!MnLogicalSize || !MbUseStorageValues)
            ThrowIt("No Logical Size");
#endif
        MnSum += FnValue;
        MnSum -= GetOldValue();
        MnAvg = MnSum / MnRunningSize;
    }
    else
    {
        MnAvg = ((MnAvg * MnLogicalSize) + FnValue) / (MnLogicalSize + 1);
        if (MnMaxTraceSize == 0 || MnMaxTraceSize > MnLogicalSize)
            MnLogicalSize++;
        MnSum = MnAvg * MnLogicalSize;
    }
}

