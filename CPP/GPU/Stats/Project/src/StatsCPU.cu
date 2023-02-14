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
    This = Other;
    Rescue();
}

RA::StatsCPU::StatsCPU(StatsCPU&& Other) noexcept
{
    //DeleteArr(MvValues);
    if (!!Other.MvValues && Other.MnStorageSize)
    {
        MvValues = Other.MvValues;
        Other.MbDelete = false;
        MnInsertIdx = Other.MnInsertIdx;
    }
    else
    {
        MnInsertIdx = 0;
        MbHadFirstInsert = false;
    }

    MnStorageSize = Other.MnStorageSize;
    MbHadFirstInsert = Other.MbHadFirstInsert;

    memcpy(MoAvgPtr, Other.MoAvgPtr, sizeof(RA::AVG));
    memcpy(MoSTOCHPtr, Other.MoSTOCHPtr, sizeof(RA::STOCH));
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

    This.Construct(MeHardware, Other.MnStorageSize, Other.MmOptions);

    if (Other.MoAvgPtr)   MoAvgPtr->CopyStats(*Other.MoAvgPtr);
    if (Other.MoSTOCHPtr) MoSTOCHPtr->CopyStats(*Other.MoSTOCHPtr);
    if (Other.MoRSIPtr)   MoRSIPtr->CopyStats(*Other.MoRSIPtr);

    Rescue();
}

void RA::StatsCPU::operator=(StatsCPU&& Other) noexcept
{
    //DeleteArr(MvValues);
    if (!!Other.MvValues && Other.MnStorageSize)
    {
        MvValues = Other.MvValues;
        Other.MbDelete = false;
        MnInsertIdx = Other.MnInsertIdx;
    }
    else
    {
        MnInsertIdx = 0;
        MbHadFirstInsert = false;
    }

    MnStorageSize = Other.MnStorageSize;
    MbHadFirstInsert = Other.MbHadFirstInsert;
}

RA::StatsCPU::StatsCPU(
    const uint FnStorageSize, 
    const xmap<EOptions, uint>& FmOptions, 
    const double FnDefaultVal)
    : RA::Stats(RA::EHardware::CPU, FnStorageSize, FmOptions, FnDefaultVal)
{
}


DHF double RA::StatsCPU::operator[](const uint IDX) const
{
    Begin();
    if (IDX >= MnStorageSize)
        ThrowIt(RED "IDX = ", MnStorageSize, " which is too big for size of\n" WHITE);
    return MvValues[IDX];
    Rescue();
}

DHF double RA::StatsCPU::Last(const uint IDX) const
{
    Begin();
    if (IDX >= MnStorageSize)
        ThrowIt(RED "IDX = ", MnStorageSize, " which is too big for size of\n" WHITE);
    return MvValues[MnStorageSize - 1 - IDX];
    Rescue();
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

