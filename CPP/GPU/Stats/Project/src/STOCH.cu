// Copyright via Apache v2 Licence [2023][Joel Leagues aka Scourge]
#if UsingMSVC
#include "STOCH.h"
#else
#include "STOCH.cuh"
#endif

RA::STOCH::STOCH(
    const double* FvValues,
    const xint    FnLogicalSize,
    const xint* FnStorageSizePtr,
    const xint* FnInsertIdxPtr)
    :
    MvValues(FvValues),
    MnLogicalSize(FnLogicalSize),
    MnStorageSizePtr(FnStorageSizePtr),
    MnInsertIdxPtr(FnInsertIdxPtr)
{
    Begin();
    // commented out because you can use RA::Stats::Construct
    //if (!*MnStorageSizePtr)
    //    ThrowIt("STOCH needs storage values");
    Rescue();
}

DXF void RA::STOCH::CopyStats(const RA::STOCH& Other)
{
    MnLogicalSize = Other.MnLogicalSize;
    MnSmallest = Other.MnSmallest;
    MnBiggest = Other.MnBiggest;
    MnSTOCH = Other.MnSTOCH;
}

DXF void RA::STOCH::Update()
{
    if (MnStorageSizePtr == nullptr || *MnStorageSizePtr == 0)
    {
        printf(RED "RSI needs storage to work\n" WHITE);
        return;
    }

    MnBiggest = -DBL_MAX; // min is big   until proven otherwise
    MnSmallest = +DBL_MAX; // max is small until proven otherwise

    const auto& LnStart = *The.MnInsertIdxPtr;
    const auto& LnStorage = *The.MnStorageSizePtr;
    const auto& LnLogic = The.MnLogicalSize;

    xint Idx = LnStart;
    for (xint i = LnStart; i < LnStart + LnLogic; i++)
    {
        // note: the first value will remove the possibility of DBL_MAX/DBL_MIN
        if (MvValues[Idx] > MnBiggest)
            MnBiggest = MvValues[Idx];
        if (MvValues[Idx] < MnSmallest)
            MnSmallest = MvValues[Idx];

        Idx = (Idx == 0) ? LnStorage - 1 : Idx - 1;
    }

    if (BxNoEntry() || (MnBiggest - MnSmallest == 0))
    {
        MnSTOCH = 50;
        return;
    }

    const auto& LnCurrent = MvValues[LnStart];
    MnSTOCH = 100 * ((LnCurrent - MnSmallest) / (MnBiggest - MnSmallest));
    //cout << "stoch val: " << MvValues[LnStart] << " : " << MnSTOCH << endl;
}

DXF void RA::STOCH::Update(const double FnValue)
{
    if (MnBiggest < FnValue)
        MnBiggest = FnValue;
    if (MnSmallest > FnValue)
        MnSmallest = FnValue;
    MnSTOCH = 100 * ((FnValue - MnSmallest) / (MnBiggest - MnSmallest));
}

DXF void RA::STOCH::SetLogicalSize(const xint FnLogicalSize)
{
    MnLogicalSize = (FnLogicalSize <= *MnStorageSizePtr) ? FnLogicalSize : *MnStorageSizePtr;
}

DXF void RA::STOCH::SetDefaultValues(const double FnDefaualt)
{
    MnBiggest = FnDefaualt;
    MnSmallest = FnDefaualt;
    MnSTOCH = 50;
}

