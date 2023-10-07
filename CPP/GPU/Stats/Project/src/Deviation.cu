
#if UsingMSVC
#include "Deviation.h"
#else
#include "Deviation.cuh"
#endif

RA::Deviation::Deviation(RA::AVG* FoAvgPtr, const EType FeType) : MoAvgPtr(FoAvgPtr), MeType(FeType)
{
    Begin();
    if (MoAvgPtr == nullptr)
        ThrowIt("Null Avg Ptr");
    Rescue();
}

DXF void RA::Deviation::Update(const double FnValue)
{
    TestRealNum(FnValue);
    auto& MoAvg = *MoAvgPtr;
    MnLastVal = MnCurrentVal;
    MnCurrentVal = FnValue;

    if (MoAvg.BxUseStorageValues())
    {
        if (MoAvg.GetRunningSize() <= 1)
            SetDefaultValues();
        else
        {
            cvar& LnStorage = MoAvg.GetStorageSize();
            cvar& LnRunSize = MoAvg.MnRunningSize;
            cvar& LnAvg = MoAvg.GetAVG();

            nvar LnInsert = MoAvg.GetInsertIdx();
            LnInsert = (LnInsert + 1 + (LnStorage - LnRunSize));
            if (LnInsert >= LnStorage)
                LnInsert = (LnInsert - LnStorage); // 20 idx - 20 size = 0 idx

            nvar LnSum = 0.0;
            nvar LnLoopsLeft = LnRunSize;

            do
            {
                cvar LnValMinusAvg =MoAvg.MvValues[LnInsert] - LnAvg;

                if (MeType == EType::SD)
                    LnSum += (LnValMinusAvg * LnValMinusAvg);
                else
                    LnSum += std::abs(LnValMinusAvg);  // << mod

                if (++LnInsert >= LnStorage)
                    LnInsert = 0;
            } while (LnLoopsLeft-- > 1);

            if (MeType == EType::SD)
                MnDeviation = sqrt(LnSum / (LnStorage - 1));
            else
                MnDeviation = LnSum / LnStorage;
        }
    }
    else
    {
        const auto LnValMinusMean = FnValue - MoAvg.GetAVG();

        if (MeType == EType::SD)
            MnSumDeviation += (LnValMinusMean * LnValMinusMean);
        else
            MnSumDeviation += std::abs(LnValMinusMean);

        if (MeType == EType::SD)
            MnDeviation = std::sqrt(MnSumDeviation / (MoAvg.GetLogicalSize() - 1));
        else
            MnDeviation = MnSumDeviation / MoAvg.GetLogicalSize();

        MnLastOffDiff = GetDirectionalOffset();
        MnSumOffset += std::abs(MnLastOffDiff);
        if (MeType == EType::SD)
            MnAvgOffset = MnSumOffset / (MoAvg.GetLogicalSize() - 1);
        else
            MnAvgOffset = MnSumOffset / MoAvg.GetLogicalSize();
    }
}

DXF void RA::Deviation::CopyStats(const Deviation& Other)
{
    MoAvgPtr = Other.MoAvgPtr;
    MeType = Other.MeType;
    MnDeviation = Other.MnDeviation;
    MnSumDeviation = Other.MnSumDeviation;
}

DXF void RA::Deviation::SetDefaultValues()
{
    MnSumDeviation = 0;
    MnDeviation = 0;
}

// -------------------------------------------------------------------------------------------------------------------------------------------------
DXF double RA::Deviation::GetDifference(const double FnCurrent, const double FnLast) const
{
    TestRealNum(FnCurrent);
    const auto LnCurrentDeviation = std::abs(FnCurrent - FnLast);
    return (MnDeviation > LnCurrentDeviation)
        ? MnDeviation - (MnDeviation - LnCurrentDeviation)
        : LnCurrentDeviation;
}

DXF double RA::Deviation::GetDifference(const double FnCurrent) const
{
    TestRealNum(FnCurrent);
    const auto LnCurrentDeviation = std::abs(FnCurrent - MnCurrentVal);
    return (MnDeviation > LnCurrentDeviation)
        ? MnDeviation - (MnDeviation - LnCurrentDeviation)
        : LnCurrentDeviation;
}

DXF double RA::Deviation::GetDifference() const
{
    const auto LnCurrentDeviation = std::abs(MnCurrentVal - MnLastVal);

    return (MnDeviation > LnCurrentDeviation)
        ? MnDeviation - (MnDeviation - LnCurrentDeviation) // current/last/avg-div = 5/4/2 >> 2 - (2 - abs(5-4)) = 2 - (1) = 1
        : LnCurrentDeviation; // current/last/avg-div = 7/4/2 >> 7 - 4 = 3 (which is gt 2) = 3
}
// -------------------------------------------------------------------------------------------------------------------------------------------------
DXF double RA::Deviation::GetDirectionalOffset(const double FnCurrent, const double FnLast) const
{
    TestRealNum(FnCurrent);
    TestRealNum(FnLast);
    return ((FnCurrent < FnLast) ? -1.0 : 1.0) * GetOffset(FnCurrent, FnLast);
}

DXF double RA::Deviation::GetDirectionalOffset(const double FnCurrent) const
{
    TestRealNum(FnCurrent);
    return ((FnCurrent < MnCurrentVal) ? -1.0 : 1.0) * GetOffset(FnCurrent);
}

DXF double RA::Deviation::GetDirectionalOffset() const
{
    return ((BxIncreasing()) ? 1.0 : -1.0) * GetOffset();
}

// -------------------------------------------------------------------------------------------------------------------------------------------------
DXF double RA::Deviation::GetOffset(const double FnCurrent, const double FnLast) const
{
    TestRealNum(FnCurrent);
    TestRealNum(FnLast);
    if (!MnDeviation)
        return 0;
    return GetDifference(FnCurrent, FnLast) / MnDeviation;
}

DXF double RA::Deviation::GetOffset(const double FnCurrent) const
{
    TestRealNum(FnCurrent);
    if (!MnDeviation)
        return 0;
    return GetDifference(FnCurrent) / MnDeviation;
}

DXF double RA::Deviation::GetOffset() const
{
    if (!MnDeviation)
        return 0;
    return GetDifference() / MnDeviation;
}
// -------------------------------------------------------------------------------------------------------------------------------------------------
DXF double RA::Deviation::GetFractinalOffset(const double FnCurrent, const double FnLast) const
{
    return GetFractional(GetOffset(FnCurrent, FnLast));
}

DXF double RA::Deviation::GetFractinalOffset(const double FnCurrent) const
{
    return GetFractional(GetOffset(FnCurrent));
}

DXF double RA::Deviation::GetFractinalOffset() const
{
    return GetFractional(GetOffset());
}
DXF double RA::Deviation::GetFractional(double FnValue)
{
    double LnReturned = 0;
    if (FnValue < 0)
    {
        FnValue *= -1; // -0.1 >> 0.1;
        FnValue += 1; // 0.1 >> 1.1
        LnReturned = 1.0 / FnValue; // 1.1 >> 0.9 >> under 10%
    }
    else
        LnReturned = FnValue + 1.0; // 0.1 >> 1.1 >> over 10%
    if (!BxIsRealNum(LnReturned))
    {
#ifdef UsingMSVC
        ThrowIt("Bad Num");
#else // UsingMSVC
        printf("Bad Num: %llf\n", LnReturned);
#endif
    }
    return LnReturned;
}
// -------------------------------------------------------------------------------------------------------------------------------------------------
