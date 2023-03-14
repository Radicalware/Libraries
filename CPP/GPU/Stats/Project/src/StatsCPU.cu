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

    The.Construct(MeHardware, Other.MnStorageSize, Other.MmOptions);
    The.SetJoinerySize(Other.MnJoinerySize);

    if (Other.MoAvgPtr)   MoAvgPtr->CopyStats(*Other.MoAvgPtr);
    if (Other.MoSTOCHPtr) MoSTOCHPtr->CopyStats(*Other.MoSTOCHPtr);
    if (Other.MoRSIPtr)   MoRSIPtr->CopyStats(*Other.MoRSIPtr);

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

    MvJoinery     = Other.MvJoinery;
    MnJoinerySize = Other.MnJoinerySize;

    MbHadFirstInsert = Other.MbHadFirstInsert;
    MmOptions = std::move(Other.MmOptions);

    MoAvgPtr   = Other.MoAvgPtr;
    MoRSIPtr   = Other.MoRSIPtr;
    MoSTOCHPtr = Other.MoSTOCHPtr;

    The.SetDeviceJoinery();
}

RA::StatsCPU::StatsCPU(
    const xint FnStorageSize, 
    const xmap<EOptions, xint>& FmOptions, 
    const double FnDefaultVal)
    : RA::Stats(RA::EHardware::CPU, FnStorageSize, FmOptions, FnDefaultVal)
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

// --------------------------------------------------------

