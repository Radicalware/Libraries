#if UsingMSVC
#include "RSI.h"
#else
#include "RSI.cuh"
#endif

RA::RSI::RSI(
    const double* FvValues,
    const xint    FnLogicalSize,
    const xint   *FnStorageSizePtr,
    const xint   *FnInsertIdxPtr)
    :
    MvValues(FvValues),
    MnLogicalSize(FnLogicalSize),
    MnStorageSizePtr(FnStorageSizePtr),
    MnInsertIdxPtr(FnInsertIdxPtr)
{
    Begin();
    // commented out because you could use RA::Stats::Construct
    //if (*MnStorageSizePtr == 0)
    //    ThrowIt("RSI needs storage to work");
    Rescue()
}

DXF double RA::RSI::GetCurvedRSI() const
{
    // Max = 75
    // Min = 25

    // good high = 70
    // good low  = 30
    if (MnRSI >= 50 && MnRSI <= 75) // standard Bull
        return MnRSI;
    else if (MnRSI > 75) // Bull Recurve
        return 75 - ((MnRSI - 75) * 3);
    else if (MnRSI >= 25 && MnRSI < 50) // Standard Bear
        return MnRSI;
    else if (MnRSI < 25) // Bear Recurve
        return 25 + ((25 - MnRSI) * 3);
    return MnRSI;
}

DXF void RA::RSI::CopyStats(const RA::RSI& Other)
{
    MnRSI = Other.MnRSI;
    MnLogicalSize = Other.MnLogicalSize;
}

DXF void RA::RSI::Update()
{
    if (MnStorageSizePtr == nullptr || *MnStorageSizePtr == 0)
    {
        printf(RED "RSI needs storage to work\n" WHITE);
        return;
    }

    double LnUpShift = 0;
    xint   LnUpTicks = 0;

    double LnDownShift = 0;
    xint   LnDownTicks = 0;

    // https://www.investopedia.com/terms/r/rsi.asp

    const auto& LnStart   = *The.MnInsertIdxPtr; 
    const auto& LnStorage = *The.MnStorageSizePtr;
    const auto& LnLogic   = The.MnLogicalSize;

    xint Idx = LnStart;
    for (xint i = LnStart; i < LnStart + LnLogic; i++)
    {
        //const xint Idx = i;

        //if (Idx > 0)
        //{
        //    LnVal2 = MvValues[Idx - 1];
        //    LnVal1 = MvValues[Idx];
        //    LnDiff = LnVal1 - LnVal2;
        //}
        //else
        //{
        //    LnVal2 = MvValues[*MnStorageSizePtr - 1];
        //    LnVal1 = MvValues[Idx];
        //    LnDiff = LnVal1 - LnVal2;
        //}

        const auto LnDiff = (Idx > 0)
            ? MvValues[Idx] - MvValues[Idx - 1]
            : MvValues[Idx] - MvValues[*MnStorageSizePtr - 1];

        if (LnDiff > 0)
        {
            LnUpShift += LnDiff;
            ++LnUpTicks;
        }
        else if (LnDiff < 0)
        {
            LnDownShift += std::abs(LnDiff);
            ++LnDownTicks;
        }
        Idx = (Idx == 0) ? LnStorage - 1 : Idx - 1;
    }

    if (!LnUpTicks && !LnDownTicks)
    {
        MnRSI = 50;
        return;
    }
    if (!LnDownTicks)
    {
        MnRSI = 100;
        return;
    }
    if (!LnUpTicks)
    {
        MnRSI = 0;
        return;
    }

    const auto LnAvgGain = LnUpShift / LnUpTicks;
    const auto LnAvgLoss = LnDownShift / LnDownTicks;

    MnRSI = 100 - (100 / (1 + (LnAvgGain / LnAvgLoss)));

    //if(LnStart >= 2)
    //    cout << MvValues[LnStart] << " : " << MvValues[LnStart - 1] << " : " << MvValues[LnStart - 2] << endl;
}

DXF void RA::RSI::SetDefaultValues(const double FnDefaualt)
{
    MnRSI = FnDefaualt;
}

