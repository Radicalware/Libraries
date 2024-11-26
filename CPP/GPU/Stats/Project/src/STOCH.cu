// Copyright via Apache v2 Licence [2023][Joel Leagues aka Scourge]
#if UsingMSVC
#include "STOCH.h"
#else
#include "STOCH.cuh"
#endif

RA::STOCH::STOCH(
    const double* FvValues,
    const   xint* FnInsertIdxPtr,
    const   xdbl* FnMinPtr,
    const   xdbl* FnMaxPtr,
    const   xint  FnStorageSize)
    :
    MvValues(FvValues),
    MnInsertIdxPtr(FnInsertIdxPtr),
    MnMinPtr(FnMinPtr),
    MnMaxPtr(FnMaxPtr),
    MnStorageSize(FnStorageSize)
{
}

DXF void RA::STOCH::CopyStats(const RA::STOCH& Other)
{
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

    cvar& LnMin = *MnMinPtr;
    cvar& LnMax = *MnMaxPtr;

    cvar& LnNewVal = MvValues[*MnInsertIdxPtr];
    const auto LnDivisor = LnMax - LnMin;
    MnSTOCH = 100.0 * ((LnNewVal - LnMin) / (LnMax - LnMin));
    if (!BxIsRealNum(MnSTOCH))
    {
        if (RA::Appx(LnDivisor, 0, 0.000001))
            MnSTOCH = 50.0;
        else
        {
            printf("1:) Bad MnSTOCH: %fl - %fl = %fl  from %fl\n", LnMax, LnMin, LnDivisor, LnNewVal);
            for (xint i = 0; i < MnStorageSize; i++)
                printf("Storage Val: %fl\n", MvValues[i]);
            printf("\n");
        }
    }
}

DXF void RA::STOCH::Update(const double FnValue)
{
    cvar& LnMin = *MnMinPtr;
    cvar& LnMax = *MnMaxPtr;

    const auto LnDivisor = LnMax - LnMin;
    MnSTOCH = 100 * ((FnValue - LnMin) / (LnMax - LnMin));
    if (!BxIsRealNum(MnSTOCH))
    {
        if (RA::Appx(LnDivisor, 0, 0.000001))
            MnSTOCH = 50.0;
        else
            printf("2:) Bad MnSTOCH: %fl - %fl = %fl\n", LnMax, LnMin, LnDivisor);
    }
}

DXF void RA::STOCH::SetDefaultValues(const double FnDefaualt)
{
    Update(FnDefaualt);
}

