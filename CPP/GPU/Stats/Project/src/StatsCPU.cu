// Copyright via Apache v2 Licence [2023][Joel Leagues aka Scourge]
#if UsingMSVC
#include "StatsCPU.h"
#else
#include "StatsCPU.cuh"
#endif

RA::StatsCPU::StatsCPU()
{
    MeHardware = EHardware::CPU;
}

RA::StatsCPU::StatsCPU(const StatsCPU& Other)
{
    Begin();
    The = Other;
    Rescue();
}

RA::StatsCPU::StatsCPU(StatsCPU&& Other) noexcept
{
    The = std::move(Other);
}

void RA::StatsCPU::operator=(const StatsCPU& Other)
{
    Begin();
    MeHardware = Other.MeHardware;
    if (!!Other.MvValues && Other.MnStorageSize)
    {
        Allocate(Other.MnStorageSize, 1);
        //memcpy(MvValues, Other.MvValues, MnStorageSize);
        MbHadFirstInsert = Other.MbHadFirstInsert;
        MnInsertIdx = Other.MnInsertIdx;
    }
    else
    {
        MnInsertIdx = 0;
        MbHadFirstInsert = false;
    }

    The.ConstructHardware(MeHardware, Other.MnStorageSize, Other.MvOptions);
    The.SetJoinerySize(Other.MoJoinery.MnSize);

    if(MnStorageSize)
        The.SetSlippageSize(Other.MoSlippage.MnSlipSize);

    if (Other.MoAvgPtr)         MoAvgPtr->CopyStats(*Other.MoAvgPtr);
    if (Other.MoSTOCHPtr)       MoSTOCHPtr->CopyStats(*Other.MoSTOCHPtr);
    if (Other.MoRSIPtr)         MoRSIPtr->CopyStats(*Other.MoRSIPtr);
    if (Other.MoNormalsPtr)     MoNormalsPtr->CopyStats(*Other.MoNormalsPtr);
    if (Other.MoStandardDeviationPtr)     MoStandardDeviationPtr->CopyStats(*Other.MoStandardDeviationPtr);
    if (Other.MoMeanAbsoluteDeviationPtr) MoMeanAbsoluteDeviationPtr->CopyStats(*Other.MoMeanAbsoluteDeviationPtr);

    The.SetDeviceJoinery();

    Rescue();
}

void RA::StatsCPU::operator=(StatsCPU&& Other) noexcept
{
    Other.MbDelete = false;
    MeHardware = Other.MeHardware;
    Clear();
    if (!!Other.MvValues && Other.MnStorageSize)
    {
        MvValues = Other.MvValues;
        MnInsertIdx = Other.MnInsertIdx;
    }
    else
    {
        MnInsertIdx = 0;
        MbHadFirstInsert = false;
    }

    MeHardware    = Other.MeHardware;
    MnStorageSize = Other.MnStorageSize;
    MnInsertIdx   = 0;
    MnMin         = Other.MnMin;
    MnMax         = Other.MnMax;

    if (Other.GetJoinerySize())
    {
        SetJoinerySize(Other.GetJoinerySize());
        for (xint i = 0; i < Other.GetJoinerySize(); i++)
            MoJoinery.MvValues[i] = Other.MoJoinery.MvValues[i];
    }

    if (Other.MoSlippage.MnSlipSize)
    {
        SetSlippageSize(Other.MoSlippage.MnSlipSize);
        for (xint i = 0; i < Other.MoSlippage.MnDataLeng; i++)
            MoSlippage.MvNums[i] = Other.MoSlippage.MvNums[i];
    }

    MbHadFirstInsert = Other.MbHadFirstInsert;
    MvOptions        = Other.MvOptions;

    MoAvgPtr        = Other.MoAvgPtr;
    MoRSIPtr        = Other.MoRSIPtr;
    MoNormalsPtr    = Other.MoNormalsPtr;
    MoSTOCHPtr      = Other.MoSTOCHPtr;
    MoStandardDeviationPtr     = Other.MoStandardDeviationPtr;
    MoMeanAbsoluteDeviationPtr = Other.MoMeanAbsoluteDeviationPtr;

    The.SetDeviceJoinery();
}

RA::StatsCPU::StatsCPU(
    const xint FnStorageSize, 
    const xvector<EStatOpt> FvOptions,
    const double FnDefaultVal)
    : RA::Stats(RA::EHardware::CPU, FnStorageSize, FvOptions, FnDefaultVal)
{
}

// --------------------------------------------------------

DHF RA::AVG& RA::StatsCPU::GetObjAVG()
{
    Begin();
    if (!MoAvgPtr)
        ThrowIt("MoAvgPtr is Null");
    return *MoAvgPtr;
    Rescue();
}

DHF RA::STOCH& RA::StatsCPU::GetObjSTOCH()
{
    Begin();
    if (!MoSTOCHPtr)
        ThrowIt("MoSTOCHPtr is Null");
    return *MoSTOCHPtr;
    Rescue();
}

DHF RA::RSI& RA::StatsCPU::GetObjRSI()
{
    Begin();
    if (!MoRSIPtr)
        ThrowIt("MoRSIPtr is Null");
    return *MoRSIPtr;
    Rescue();
}

DHF RA::Normals& RA::StatsCPU::GetObjNormals()
{
    Begin();
    if (!MoNormalsPtr)
        ThrowIt("MoNormalsPtr is Null");
    return *MoNormalsPtr;
    Rescue();
}

DHF RA::Deviation& RA::StatsCPU::GetObjStandardDeviation()
{
    Begin();
    if (!MoStandardDeviationPtr)
        ThrowIt("MoStandardDeviationPtr is Null");
    return *MoStandardDeviationPtr;
    Rescue();
}


DHF RA::Deviation& RA::StatsCPU::GetObjMeanAbsoluteDeviation()
{
    Begin();
    if (!MoMeanAbsoluteDeviationPtr)
        ThrowIt("MoMeanAbsoluteDeviationPtr is Null");
    return *MoMeanAbsoluteDeviationPtr;
    Rescue();
}

// --------------------------------------------------------

DHF const RA::AVG& RA::StatsCPU::GetObjAVG() const
{
    Begin();
    if (!MoAvgPtr)
        ThrowIt("MoAvgPtr is Null");
    return *MoAvgPtr;
    Rescue();
}

DHF const RA::STOCH& RA::StatsCPU::GetObjSTOCH() const
{
    Begin();
    if (!MoSTOCHPtr)
        ThrowIt("MoSTOCHPtr is Null");
    return *MoSTOCHPtr;
    Rescue();
}

DHF const RA::RSI& RA::StatsCPU::GetObjRSI() const
{
    Begin();
    if (!MoRSIPtr)
        ThrowIt("MoRSIPtr is Null");
    return *MoRSIPtr;
    Rescue();
}

DHF const RA::Normals& RA::StatsCPU::GetObjNormals() const
{
    Begin();
    if (!MoNormalsPtr)
        ThrowIt("MoNormalsPtr is Null");
    return *MoNormalsPtr;
    Rescue();
}

DHF const RA::Deviation& RA::StatsCPU::GetObjStandardDeviation() const
{
    Begin();
    if (!MoStandardDeviationPtr)
        ThrowIt("MoStandardDeviationPtr is Null");
    return *MoStandardDeviationPtr;
    Rescue();
}


DHF const RA::Deviation& RA::StatsCPU::GetObjMeanAbsoluteDeviation() const
{
    Begin();
    if (!MoMeanAbsoluteDeviationPtr)
        ThrowIt("MoMeanAbsoluteDeviationPtr is Null");
    return *MoMeanAbsoluteDeviationPtr;
    Rescue();
}

// --------------------------------------------------------

DHF const RA::AVG& RA::StatsCPU::AVG() const
{
    Begin();
    if (!MoAvgPtr)
        ThrowIt("MoAvgPtr is Null");
    return *MoAvgPtr;
    Rescue();
}

DHF const RA::STOCH& RA::StatsCPU::STOCH() const
{
    Begin();
    if (!MoSTOCHPtr)
        ThrowIt("MoSTOCHPtr is Null");
    return *MoSTOCHPtr;
    Rescue();
}

DHF const RA::RSI& RA::StatsCPU::RSI() const
{
    Begin();
    if (!MoRSIPtr)
        ThrowIt("MoRSIPtr is Null");
    return *MoRSIPtr;
    Rescue();
}

DHF const RA::Normals& RA::StatsCPU::Normals() const
{
    Begin();
    if (!MoNormalsPtr)
        ThrowIt("MoNormalsPtr is Null");
    return *MoNormalsPtr;
    Rescue();
}

DHF const RA::Deviation& RA::StatsCPU::SD() const
{
    Begin();
    if (!MoStandardDeviationPtr)
        ThrowIt("MoStandardDeviationPtr is Null");
    return *MoStandardDeviationPtr;
    Rescue();
}

DHF const RA::Deviation& RA::StatsCPU::MAD() const
{
    Begin();
    if (!MoMeanAbsoluteDeviationPtr)
        ThrowIt("MoMeanAbsoluteDeviationPtr is Null");
    return *MoMeanAbsoluteDeviationPtr;
    Rescue();
}

// --------------------------------------------------------


DHF RA::AVG& RA::StatsCPU::AVG()
{
    Begin();
    if (!MoAvgPtr)
        ThrowIt("MoAvgPtr is Null");
    return *MoAvgPtr;
    Rescue();
}

DHF RA::STOCH& RA::StatsCPU::STOCH()
{
    Begin();
    if (!MoSTOCHPtr)
        ThrowIt("MoSTOCHPtr is Null");
    return *MoSTOCHPtr;
    Rescue();
}

DHF RA::RSI& RA::StatsCPU::RSI()
{
    Begin();
    if (!MoRSIPtr)
        ThrowIt("MoRSIPtr is Null");
    return *MoRSIPtr;
    Rescue();
}

DHF RA::Normals& RA::StatsCPU::Normals()
{
    Begin();
    if (!MoNormalsPtr)
        ThrowIt("MoNormalsPtr is Null");
    return *MoNormalsPtr;
    Rescue();
}


DHF RA::Deviation& RA::StatsCPU::SD()
{
    Begin();
    if (!MoStandardDeviationPtr)
        ThrowIt("MoStandardDeviationPtr is Null");
    return *MoStandardDeviationPtr;
    Rescue();
}

DHF RA::Deviation& RA::StatsCPU::MAD()
{
    Begin();
    if (!MoMeanAbsoluteDeviationPtr)
        ThrowIt("MoMeanAbsoluteDeviationPtr is Null");
    return *MoMeanAbsoluteDeviationPtr;
    Rescue();
}

DHF double RA::StatsCPU::Get(const RA::EStatOpt FeOption) const
{
    Begin();
    switch (FeOption)
    {
    case RA::EStatOpt::Literal: return ValueFromEnd();
    case RA::EStatOpt::AVG: return GetAVG();
    case RA::EStatOpt::RSI: return RSI().GetScaledRSI();
    case RA::EStatOpt::STOCH: return STOCH().GetScaledSTOCH();
    case RA::EStatOpt::MAD: return The.MAD().GetOffset();
    case RA::EStatOpt::SD: return The.SD().GetOffset();
    default:
            ThrowIt("No Option: ", (xint)FeOption);
    }
    Rescue();
}