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
        MvOptions = Other.MvOptions;
        Allocate(Other.MnStorageSize, Other.MvValues);
        //memcpy(MvValues, Other.MvValues, MnStorageSize);
        MbHadFirstInsert = Other.MbHadFirstInsert;
        MnInsertIdx = Other.MnInsertIdx;
        CreateObjs(MvOptions);
    }
    else
    {
        The.ConstructHardware(MeHardware, Other.MnStorageSize, Other.MvOptions);
        MnInsertIdx = 0;
        MbHadFirstInsert = false;
    }

    The.SetJoinerySize(Other.MoJoinery.MnSize);

    if(MnStorageSize)
        The.SetSlippageSize(Other.MoSlippage.MnSlipSize);

    if (Other.MoAvgPtr)         MoAvgPtr->CopyStats(*Other.MoAvgPtr);
    if (Other.MoZeaPtr)         MoZeaPtr->CopyStats(*Other.MoZeaPtr);
    if (Other.MoOmahaPtr)       MoOmahaPtr->CopyStats(*Other.MoOmahaPtr);
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
    MoZeaPtr        = Other.MoZeaPtr;
    MoOmahaPtr      = Other.MoOmahaPtr;
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


DHF RA::AVG& RA::StatsCPU::AVG()
{
    Begin();
    if (!MoAvgPtr)
        ThrowIt("MoAvgPtr is Null");
    return *MoAvgPtr;
    Rescue();
}

DHF RA::ZEA& RA::StatsCPU::ZEA()
{
    Begin();
    if (!MoZeaPtr)
        ThrowIt("MoZeaPtr is Null");
    return *MoZeaPtr;
    Rescue();
}

DHF RA::Omaha& RA::StatsCPU::Omaha()
{
    Begin();
    if (!MoOmahaPtr)
        ThrowIt("MoOmahaPtr is Null");
    return *MoOmahaPtr;
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
    case RA::EStatOpt::Literal: return GetBack();
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