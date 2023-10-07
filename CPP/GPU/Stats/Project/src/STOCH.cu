// Copyright via Apache v2 Licence [2023][Joel Leagues aka Scourge]
#if UsingMSVC
#include "STOCH.h"
#else
#include "STOCH.cuh"
#endif

RA::STOCH::STOCH(
    const double* FvValues,
    const   xint* FnInsertIdxPtr,
    const   xint  FnStorageSize)
    :
    MvValues(FvValues),
    MnInsertIdxPtr(FnInsertIdxPtr),
    MnStorageSize(FnStorageSize)
{
    Begin();
    Rescue();
}

DXF void RA::STOCH::CopyStats(const RA::STOCH& Other)
{
    MnSmallest = Other.MnSmallest;
    MnBiggest = Other.MnBiggest;
    MnSTOCH = Other.MnSTOCH;
}

DXF void RA::STOCH::Update()
{
    if (MvValues == nullptr || MnStorageSize == 0)
    {
        printf(RED "STOCH needs storage to work\n" WHITE);
        return;
    }

    MnRunningSize++;
    if (MnRunningSize >= MnStorageSize)
        MnRunningSize = MnStorageSize;

    MnBiggest  = -DBL_MAX; // min is big   until proven otherwise
    MnSmallest = +DBL_MAX; // max is small until proven otherwise

    auto LnInsert = *The.MnInsertIdxPtr + 1;
    if (LnInsert >= MnStorageSize) LnInsert = 0;
    auto LnLoopsLeft = MnRunningSize - 1;

    do
    {
        const auto LnValue = MvValues[LnInsert];
        if (MnBiggest < LnValue)
            MnBiggest = LnValue;
        if (MnSmallest > LnValue)
            MnSmallest = LnValue;

        if (++LnInsert >= MnStorageSize)
            LnInsert = 0;
    } 
    while (LnLoopsLeft-- > 1); // you don't count yourself

    if (BxNoEntry())
    {
        MnSTOCH = 50;
        return;
    }

    cvar& LnVal = MvValues[*MnInsertIdxPtr];
    MnSTOCH = 100 * ((LnVal - MnSmallest) / (MnBiggest - MnSmallest));
}

DXF void RA::STOCH::Update(const double FnValue)
{
    if (MnBiggest < FnValue)
        MnBiggest = FnValue;
    if (MnSmallest > FnValue)
        MnSmallest = FnValue;
    MnSTOCH = 100 * ((FnValue - MnSmallest) / (MnBiggest - MnSmallest));
}

DXF void RA::STOCH::SetDefaultValues(const double FnDefaualt)
{
    MnBiggest = FnDefaualt;
    MnSmallest = FnDefaualt;
    MnSTOCH = 50;
}

