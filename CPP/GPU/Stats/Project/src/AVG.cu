#if UsingMSVC
#include "AVG.h"
#else
#include "AVG.cuh"
#endif

RA::AVG::AVG(
    const double* FvValues,
    const uint    FnLogicalSize,
    const uint   *FnStorageSizePtr,
    const uint   *FnInsertIdxPtr)
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

DXF void RA::AVG::Update()
{
#if UsingMSVC
    if (!MnLogicalSize || !MbUseStorageValues)
        ThrowIt("No Logical Size");
#endif

    if (MnAvg == 0 && *MnInsertIdxPtr == 0)
        MnAvg = MvValues[*MnInsertIdxPtr];
    else
    {
        //const uint LnStartIdx = (*MnInsertIdxPtr >= MnLogicalSize)
        //    ? *MnInsertIdxPtr - MnLogicalSize
        //    : *MnStorageSizePtr - (MnLogicalSize - *MnInsertIdxPtr);

        //const auto& LnReplacedVal = MvValues[LnStartIdx];
        //const auto& LnInertVal    = MvValues[*MnInsertIdxPtr];
        //const auto  LnSumUnderOne = MnAvg * (MnLogicalSize - 1);
        //MnAvg = ((LnSumUnderOne + (LnInertVal - (LnReplacedVal - MnAvg))) / MnLogicalSize);


        const auto& LnStart     = *This.MnInsertIdxPtr;
        const auto& LnStorage   = *This.MnStorageSizePtr;
        const auto& LnLogic     =  This.MnLogicalSize;

        uint Idx = LnStart;
        MnSum = 0;
        for (uint i = LnStart; i < LnStart + LnLogic; i++)
        {
            const auto& LnVal = MvValues[Idx];
            MnSum += LnVal;
            Idx = (Idx == 0) ? LnStorage - 1 : Idx - 1;
        }
        MnAvg = MnSum / This.MnLogicalSize;
    }
}

DXF void RA::AVG::Update(const double FnValue)
{
    if (MbUseStorageValues)
        return Update();
    MnAvg = ((MnAvg * MnLogicalSize) + FnValue) / (MnLogicalSize + 1);
    MnLogicalSize++;
    MnSum = MnAvg * MnLogicalSize;
}

DXF void RA::AVG::SetDefaultValues(const double FnDefaualt)
{
    MnAvg = FnDefaualt;
}

