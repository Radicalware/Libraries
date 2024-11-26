#if UsingMSVC
#include "RSI.h"
#else
#include "RSI.cuh"
#endif

RA::RSI::RSI(
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
}

DXF void RA::RSI::Update()
{
    if (MvValues == nullptr || MnStorageSize == 0)
    {
        printf(RED "RSI needs storage to work\n" WHITE);
        return;
    }

    MnRunningSize++;
    if (MnRunningSize >= MnStorageSize)
        MnRunningSize = MnStorageSize;

    if (MnRunningSize <= 1)
    {
        MnRSI = 50;
        return;
    }

    double LnUpShift = 0;
    xint   LnUpTicks = 0;

    double LnDownShift = 0;
    xint   LnDownTicks = 0;

    // https://www.investopedia.com/terms/r/rsi.asp

    nvar LnInsert = *The.MnInsertIdxPtr;
    LnInsert = ((LnInsert + 2) + (MnStorageSize - MnRunningSize));
    if (LnInsert >= (MnStorageSize))
        LnInsert = (LnInsert - MnStorageSize); // 20 idx - 20 size = 0 idx

    nvar LnLast = (LnInsert > 0) ? LnInsert - 1 : MnStorageSize - 1;

    nvar LnLoopsLeft = (MnRunningSize - 1);
    
    do
    {
        cvar& LnCurrentVal = MvValues[LnInsert];
        cvar& LnLastVal    = MvValues[LnLast];
        cvar  LnDiff       = LnCurrentVal - LnLastVal;

        if (LnDiff < 0)
        {
            ++LnDownTicks;
            LnDownShift += std::abs(LnDiff);
        }
        else if (LnDiff > 0)
        {
            ++LnUpTicks;
            LnUpShift += LnDiff;
        }

        if (++LnInsert >= MnStorageSize) // update
            LnInsert = 0;
        LnLast = (LnInsert > 0) ? LnInsert - 1 : MnStorageSize - 1;
    }
    while (LnLoopsLeft-- > 1); // for RSI because we do a compare

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

    const auto LnAvgGain = LnUpShift / (double)LnUpTicks;
    const auto LnAvgLoss = LnDownShift / (double)LnDownTicks;

    MnRSI = 100.0 - (100.0 / (1.0 + (LnAvgGain / LnAvgLoss)));

    //if(LnStart >= 2)
    //    cout << MvValues[LnStart] << " : " << MvValues[LnStart - 1] << " : " << MvValues[LnStart - 2] << endl;
}

DXF void RA::RSI::SetDefaultValues(const double FnDefaualt)
{
    MnRSI = FnDefaualt;
}

