
#if UsingMSVC
#include "Deviation.h"
#else
#include "Deviation.cuh"
#endif

DXF void RA::Deviation::UpdateStandardDeviation(const double FnValue)
{
    TestRealNum(FnValue);
    auto& MoAvg = *MoAvgPtr;
    MnLastVal = MnCurrentVal;
    MnCurrentVal = FnValue;
    if (MoAvg.GetLogicalSize() <= 1)
        MnLastVal = FnValue;

    if (MoAvg.MbUseStorageValues)
    {
#if UsingMSVC
        if (!MoAvg.MnLogicalSize || !MoAvg.MbUseStorageValues)
            ThrowIt("No Logical Size");
#endif

        if (MoAvg.MnAvg == 0 && *MoAvg.MnInsertIdxPtr == 0)
            SetDefaultValues();
        else
        {
            const auto& LnStorage = *MoAvg.MnStorageSizePtr;
            const auto  LnLogicSize = MoAvg.MnRunningSize;
            const auto& LnStart = *MoAvg.MnInsertIdxPtr;
            const auto  LnAvg = MoAvg.GetAVG();

            // SquareRoot ( (Sigma (Val - Avg)^2) / (MnRunningSize - 1) )
            xint Idx = (LnStart == 0) ? LnStorage - 1 : LnStart - 1;
            auto LnSum = 0.0;
            LnSum = std::pow((FnValue - LnAvg), 2);
            for (xint i = LnStart + 1; i < LnStart + LnLogicSize; i++)
            {
                const auto& LnVal = MoAvg.MvValues[Idx];
                LnSum += std::pow((LnVal - LnAvg), 2);
                Idx = (Idx == 0) ? LnStorage - 1 : Idx - 1;
            }
            MnDeviation = sqrt(LnSum / (LnLogicSize - 1));
        }
    }
    else
    {
        MnSumDeviation += std::pow((MnCurrentVal - MoAvg.GetAVG()), 2);
        if (MoAvg.GetLogicalSize() == 2)
            MnSumDeviation += MnSumDeviation;

        MnDeviation = std::sqrt(MnSumDeviation / MoAvg.GetLogicalSize());

        MnLastOffDiff = GetDirectionalOffset();
        MnSumOffset += std::abs(MnLastOffDiff);;
        MnAvgOffset = MnSumOffset / MoAvg.GetLogicalSize();
    }
}

DXF void RA::Deviation::UpdateMeanAbsoluteDeviation(const double FnCurrent)
{
    TestRealNum(FnCurrent);
    auto& MoAvg = *MoAvgPtr;
    MnLastVal = MnCurrentVal;
    MnCurrentVal = FnCurrent;
    if (MoAvg.GetLogicalSize() <= 1)
        MnLastVal = FnCurrent;

    if (MoAvg.BxUseStorageValues())
    {
#if UsingMSVC
        if (!MoAvg.GetLogicalSize() || !MoAvg.BxUseStorageValues())
            ThrowIt("No Logical Size");
#endif
        if (MoAvg.GetAVG() == 0 && MoAvg.GetInsertIdx() == 0)
            SetDefaultValues();
        else
        {
            const auto& LnStorage = *MoAvg.MnStorageSizePtr;
            const auto  LnLogicSize = MoAvg.MnRunningSize;
            const auto& LnStart = *MoAvg.MnInsertIdxPtr;
            const auto  LnAvg = MoAvg.GetAVG();

            // Sigma(abs(Val - Avg)) / MnRunningSize
            xint Idx = (LnStart == 0) ? LnStorage - 1 : LnStart - 1;
            auto LnSum = 0.0;
            LnSum = std::abs(FnCurrent - LnAvg);
            for (xint i = LnStart + 1; i < LnStart + LnLogicSize; i++)
            {
                const auto& LnVal = MoAvg.MvValues[Idx];
                LnSum += std::abs(LnVal - LnAvg);
                Idx = (Idx == 0) ? LnStorage - 1 : Idx - 1;
            }
            MnDeviation = (LnSum / LnLogicSize);
        }
    }
    else
    {
        MnSumDeviation += std::abs(MnCurrentVal - MoAvg.GetAVG());
        MnDeviation = MnSumDeviation / MoAvg.GetLogicalSize();

        MnLastOffDiff = GetDirectionalOffset();
        MnSumOffset += std::abs(MnLastOffDiff);
        MnAvgOffset = MnSumOffset / MoAvg.GetLogicalSize();
    }
}

RA::Deviation::Deviation(RA::AVG* FoAvgPtr, const EType FeType) : MoAvgPtr(FoAvgPtr), MeType(FeType)
{
    Begin();
    if (MoAvgPtr == nullptr)
        ThrowIt("Null Avg Ptr");
    Rescue();
}

DXF void RA::Deviation::Update(const double FnValue)
{
    switch (MeType)
    {
    case RA::Deviation::EType::None:
        printf("Error: None Type for Deviation");
    case RA::Deviation::EType::MAD:
        return UpdateMeanAbsoluteDeviation(FnValue);
    case RA::Deviation::EType::SD:
        return UpdateStandardDeviation(FnValue);
    default:
        return;
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
