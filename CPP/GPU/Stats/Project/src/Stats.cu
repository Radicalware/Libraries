#if UsingMSVC
#include "Stats.h"
#else
#include "Stats.cuh"
#endif
#include "Memory.h"

// Copyright via Apache v2 Licence [2023][Joel Leagues aka Scourge]

RA::Stats::Stats()
{
}

RA::Stats::~Stats()
{
    if (MbDelete)
        Clear();
}

void RA::Stats::Clear()
{
    Begin();

    ClearStorageRequriedObjs();
    ClearJoinery();
    ClearSlippage();

    if (MeHardware == EHardware::GPU) {
        CudaDelete(MvValues);
    }
    else if (MeHardware == EHardware::CPU) {
        HostDelete(MvValues);
    }
    else
        ThrowIt("EHardware (GPU/CPU) Option Not Given");

    Rescue();
}

void RA::Stats::ClearJoinery()
{
    Begin();
    if (MeHardware == EHardware::GPU)
    {
        if (MoJoinery.MvValues)
            ThrowIt("Joinery not supported on the GPU atm");
    }
    else if (MeHardware == EHardware::CPU)
    {
        HostDelete(MoJoinery.MvValues);
    }
    else
        ThrowIt("EHardware (GPU/CPU) Option Not Given");
    Rescue();
}

void RA::Stats::ClearSlippage()
{
    Begin();
    if (MeHardware == EHardware::CPU) {
        HostDelete(MoSlippage.MvNums)
    }
    else if (MeHardware == EHardware::GPU) {
        CudaDelete(MoSlippage.MvNums);
    }
    Rescue();
}

void RA::Stats::ClearStorageRequriedObjs()
{
    Begin();
    switch (MeHardware)
    {
    case EHardware::GPU:
    {
        CudaDelete(MoAvgPtr);
        CudaDelete(MoSTOCHPtr);
        CudaDelete(MoRSIPtr);
        CudaDelete(MoStandardDeviationPtr);
        CudaDelete(MoMeanAbsoluteDeviationPtr);
        break;
    }
    case EHardware::CPU:
    {
        HostDelete(MoAvgPtr);
        HostDelete(MoSTOCHPtr);
        HostDelete(MoRSIPtr);
        HostDelete(MoStandardDeviationPtr);
        HostDelete(MoMeanAbsoluteDeviationPtr);
        break;
    }
    default:
        ThrowIt("EHardware (GPU/CPU) Option Not Given");
    }
    Rescue();
}


void RA::Stats::CreateObjs(const xvector<EStatOpt>& FvOptions)
{
    Begin();
    MvOptions = FvOptions;
    ClearStorageRequriedObjs();
    switch (MeHardware)
    {
    case RA::EHardware::CPU:
    {
        for (const EStatOpt& LeOpt : MvOptions)
        {
            if (LeOpt == EStatOpt::AVG)
                MoAvgPtr = new AVG(MvValues, &MnInsertIdx, MnStorageSize);
            else if (LeOpt == EStatOpt::STOCH)
                MoSTOCHPtr = new STOCH(MvValues, &MnInsertIdx, MnStorageSize);
            else if (LeOpt == EStatOpt::RSI)
                MoRSIPtr = new RSI(MvValues, &MnInsertIdx, MnStorageSize);
            else if (LeOpt == EStatOpt::SD)
                MoStandardDeviationPtr = new Deviation(MoAvgPtr, Deviation::EType::SD);
            else if (LeOpt == EStatOpt::MAD)
                MoMeanAbsoluteDeviationPtr = new Deviation(MoAvgPtr, Deviation::EType::MAD);
        }
        break;
    }
#ifndef UsingMSVC
    case RA::EHardware::GPU:
    {
        for (const EStatOpt& LeOpt : MvOptions)
        {
            if (LeOpt == EStatOpt::AVG)
                MoAvgPtr = RA::Host::AllocateObjOnDevice<AVG>(MvValues, &MnInsertIdx, MnStorageSize);
            else if (LeOpt == EStatOpt::STOCH)
                MoSTOCHPtr = RA::Host::AllocateObjOnDevice<STOCH>(MvValues, &MnInsertIdx, MnStorageSize);
            else if (LeOpt == EStatOpt::RSI)
                MoRSIPtr = RA::Host::AllocateObjOnDevice<RSI>(MvValues, &MnInsertIdx, MnStorageSize);
            else if (LeOpt == EStatOpt::SD)
                MoStandardDeviationPtr = RA::Host::AllocateObjOnDevice<Deviation>(MoAvgPtr, Deviation::EType::SD);
            else if (LeOpt == EStatOpt::MAD)
                MoMeanAbsoluteDeviationPtr = RA::Host::AllocateObjOnDevice<Deviation>(MoAvgPtr, Deviation::EType::MAD);
        }
        break;
    }
#endif
    default:
        ThrowIt("EHardware (GPU/CPU) Option Not Given");
    }
    Rescue();
}

void RA::Stats::Allocate(const xint FnStorageSize, const double FnDefaultVal)
{
    Begin();
    Clear();
    switch (MeHardware)
    {
    case RA::EHardware::CPU:
    {
        MnStorageSize = FnStorageSize;
        MnInsertIdx = 0;
        MbHadFirstInsert = false;
        if (MnStorageSize)
        {
            DeleteArr(MvValues);
            MvValues = new double[FnStorageSize + 1];
            for (auto* Ptr = MvValues; Ptr < MvValues + FnStorageSize; Ptr++)
                *Ptr = FnDefaultVal;
        }
        break;
    }

#ifndef UsingMSVC
    case RA::EHardware::GPU:
    {
        MnStorageSize = FnStorageSize;
        MnInsertIdx = 0;
        MbHadFirstInsert = false;
        if (MnStorageSize)
        {
            CudaDelete(MvValues);
            auto Ptr = RA::MakeShared<double[]>(MnStorageSize + 1);
            for (auto& Val : Ptr)
                Val = FnDefaultVal;
            MvValues = RA::Host::AllocateArrOnDevice<double>(Ptr.Raw(), RA::Allocate(MnStorageSize, sizeof(double)));
        }
        break;
    }
#endif
    default:
        ThrowIt("EHardware (GPU/CPU) Option Not Given");
    }
    SetDeviceJoinery();
    Rescue();
}

void RA::Stats::Construct(
    const xint FnStorageSize,
    const xvector<EStatOpt>& FvOptions,
    const double FnDefaultVal)
{
    Begin();

    MvOptions = FvOptions;
    if (!!FvOptions)
    {
        Clear();
    }

    MnStorageSize = FnStorageSize;

    Allocate(MnStorageSize, FnDefaultVal);
    CreateObjs(MvOptions);

    if (MeHardware == EHardware::CPU && FnDefaultVal)
        SetDefaultValues(FnDefaultVal);

    MnInsertIdx = 0;
    MbHadFirstInsert = false;

    Rescue();
}


void RA::Stats::ConstructHardware(
    const EHardware FeHardware,
    const xint FnStorageSize,
    const xvector<EStatOpt>& FvOptions,
    const double FnDefaultVal)
{
    Begin();
    MeHardware = FeHardware;
    The.Construct(FnStorageSize, FvOptions, FnDefaultVal);
    Rescue();
}

RA::Stats::Stats(
    const EHardware FeHardware,
    const xint FnStorageSize,
    const xvector<EStatOpt>& FvOptions, // Options <> Logical Size
    const double FnDefaultVal)
{
    Begin();
    The.ConstructHardware(FeHardware, FnStorageSize, FvOptions, FnDefaultVal);
    Rescue();
}


void RA::Stats::SetStorageSizeZero(const double FnDefaultVal)
{
    Begin();
    if (MvValues)
    {
        if(MeHardware == EHardware::GPU){
            CudaDelete(MvValues);
        }
        else{
            HostDelete(MvValues);
        }
    }
    MnStorageSize = 0;
    SRef(MoAvgPtr).SetDefaultValues(FnDefaultVal);

    ClearStorageRequriedObjs();

    MnInsertIdx = 0;
    MbHadFirstInsert = false;
    Rescue();
}

DXF void RA::Stats::SetDefaultValues(const double FnDefaultVal)
{
    SRef(MoAvgPtr).SetDefaultValues(FnDefaultVal);
    SRef(MoSTOCHPtr).SetDefaultValues(FnDefaultVal);
    SRef(MoRSIPtr).SetDefaultValues();
    SRef(MoStandardDeviationPtr).SetDefaultValues();
    SRef(MoMeanAbsoluteDeviationPtr).SetDefaultValues();
}

void RA::Stats::SetJoinerySize(const xint FCount)
{
    if (FCount <= 1)
        return;
    MoJoinery.MnSize = FCount;
    MnInsertIdx = 0;
    MbHadFirstInsert = false;
    ClearJoinery();

    switch (MeHardware)
    {
    case RA::EHardware::CPU:
    {
        if (MoJoinery.MnSize)
        {
            //MoJoinery.MvChunks = new Chunk[MnStorageSize+1];
            //GET(LoJoinData, MoJoinery.MoData);
            
            MoJoinery.MnSum = 0;
            MoJoinery.MnIdx = 0;
            MoJoinery.MnSize = MoJoinery.MnSize;
            MoJoinery.MvValues = new double[MoJoinery.MnSize + 1];
            for (auto* Ptr2 = MoJoinery.MvValues; Ptr2 <= MoJoinery.MvValues + MoJoinery.MnSize; Ptr2++)
                *Ptr2 = 0;
        }
        else
            Clear();
        break;
    }
#ifndef UsingMSVC
    case RA::EHardware::GPU:
    {
        if (MoJoinery.MnSize)
        {
            ThrowIt("Not supported on the GPU atm");
        }
        else
            Clear();
        break;
    }
#endif
    default:
        ThrowIt("EHardware (GPU/CPU) Option Not Given");
    }
}

void RA::Stats::SetSlippageSize(const xint FCount)
{
    Begin();
    if (FCount <= 1)
        return;

    ClearSlippage();
    if (!FCount)
    {
        if (MoSlippage.MvNums)
        {
            if (MeHardware == EHardware::CPU) {
                HostDelete(MoSlippage.MvNums)
            }
            else if (MeHardware == EHardware::GPU) {
                CudaDelete(MoSlippage.MvNums);
            }
            else
                ThrowIt("No Hardware Set");
        }
        MoSlippage.MnDataLeng = 0;
        MoSlippage.MnSlipSize = 0;
        return;
    }
    if (!MvValues)
        return;
    MoSlippage.MeHardware = MeHardware;
    MoSlippage.MnDataLeng = (FCount * MnStorageSize);
    MoSlippage.MnSlipSize = FCount;
    MoSlippage.MvNums = new double[MoSlippage.MnDataLeng+1];
    for (auto Ptr = MoSlippage.MvNums; Ptr != (MoSlippage.MvNums + MoSlippage.MnDataLeng); Ptr++)
        *Ptr = 0;
    Rescue();
}


DXF void RA::Stats::SetMaxTraceSize(const xint FSize)
{
    MnMaxTraceSize = FSize;
    SRef(MoAvgPtr).SetMaxTraceSize(FSize);
    SRef(MoStandardDeviationPtr).SetMaxTraceSize(FSize);
    SRef(MoMeanAbsoluteDeviationPtr).SetMaxTraceSize(FSize);
}

DXF double RA::Stats::operator[](const xint IDX) const
{
    if (!MnStorageSize && !IDX)
        return MnValue;
    if (IDX >= MnStorageSize)
#ifdef UsingMSVC
        ThrowIt(RED "IDX = ", MnStorageSize, " which is too big for size of" WHITE);
#else // UsingMSVC
        printf(RED "IDX = %llu which is too big for size of\n" WHITE, MnStorageSize);
#endif
    if (MnInsertIdx >= IDX)
        return MvValues[MnInsertIdx - IDX];
    const auto LnRelIDX = MnInsertIdx + MnStorageSize - IDX; // 0 + 5 - 1 == 5 - 1 == idx 4 (of size o5)
    return MvValues[LnRelIDX];
}

DXF double RA::Stats::Former(const xint IDX) const
{
    const auto LnIdx = IDX + 1;
    if (LnIdx >= MnStorageSize)
#ifdef UsingMSVC
        ThrowIt(RED "IDX = ", MnStorageSize, " which is too big for size of" WHITE);
#else // UsingMSVC
        printf(RED "IDX = %llu which is too big for size of\n" WHITE, MnStorageSize);
#endif
    const auto LnRelIdx = MnInsertIdx + 1 + LnIdx; // where 0 is the start
    if (LnRelIdx >= MnStorageSize)
        return MvValues[LnRelIdx - MnStorageSize];
    return MvValues[LnRelIdx];
}

DXF void RA::Stats::operator<<(const double FnValue)
{
    if (MbHadFirstInsert && !MoJoinery.MnSize)
    {
        // newer numbers are behind
        // oldest number is +1 ahead but increasing in newness
        if (++MnInsertIdx >= MnStorageSize)
            MnInsertIdx = 0;
    }

    MnValue = FnValue;

    if (MoJoinery.MnSize)
    {
        if (MbHadFirstInsert && MoJoinery.MnIdx == 0 && ++MnInsertIdx >= MnStorageSize)
            MnInsertIdx = 0;

        MoJoinery.InsertNum(FnValue);

        if (MvValues)
            MvValues[MnInsertIdx] = MoJoinery.MnSum;

        MnValue = MoJoinery.MnSum;
        if (++MoJoinery.MnIdx >= MoJoinery.MnSize)
            MoJoinery.MnIdx = 0;
    }

    if (MoSlippage.MnDataLeng)
    {
        auto LnSlipIdx = 0;
        auto LnInsertIdx = 0;
        auto LnMainIdx = MnInsertIdx;
        cvar LnNewVal = (MoJoinery.MvValues) ? MoJoinery.MnSum : FnValue;

        do
        {
            if (LnSlipIdx == MoSlippage.MnDataLeng)
                MoSlippage.MvNums[0] = MoSlippage.MvNums[1];
            else if (LnSlipIdx == (MoSlippage.MnDataLeng - 1))
                MoSlippage.MvNums[LnSlipIdx] = MoSlippage.MvNums[0];
            else
                MoSlippage.MvNums[LnSlipIdx] = MoSlippage.MvNums[LnSlipIdx + 1];
            
            if (++LnInsertIdx == (MoSlippage.MnSlipSize))
            {
                LnInsertIdx = 0;
                if (++LnMainIdx >= MnStorageSize)
                    LnMainIdx = 0;
                const auto& LnDbgSlipNum = MoSlippage.MvNums[LnSlipIdx];
                MvValues[LnMainIdx] = MoSlippage.MvNums[LnSlipIdx];
            }

            ++LnSlipIdx;
            if (LnSlipIdx == MoSlippage.MnDataLeng)
                LnSlipIdx = 0;
            else if (LnSlipIdx == (MoSlippage.MnDataLeng+1))
                LnSlipIdx = 1;
        } 
        while (LnSlipIdx != 0);

        MoSlippage.MvNums[MoSlippage.MnDataLeng - 1] = LnNewVal;
        MvValues[MnInsertIdx] = LnNewVal;
        MnValue = LnNewVal;
    }
    
    if (MvValues)
        MvValues[MnInsertIdx] = MnValue;

    if (!MnStorageSize || !MvValues)
    {
        SRef(MoAvgPtr).Update(MnValue);
        SRef(MoStandardDeviationPtr).Update(MnValue);
        SRef(MoMeanAbsoluteDeviationPtr).Update(MnValue);
        SRef(MoSTOCHPtr).Update(MnValue);
        return;
    }
 

    if (MnInsertIdx != GetInsertIdx())
        printf(RED __CLASS__ " >> Bad Index Alignment\n" WHITE);

    SRef(MoAvgPtr).Update(MnValue);
    SRef(MoStandardDeviationPtr).Update(MnValue);
    SRef(MoMeanAbsoluteDeviationPtr).Update(MnValue);
    SRef(MoRSIPtr).Update();
    SRef(MoSTOCHPtr).Update();

    MbHadFirstInsert = true;
}

DXF void RA::Stats::Reset()
{
    SetAllValues(0, false);
    if(MoAvgPtr)
        (*MoAvgPtr).ResetRunningSize();
}

DXF void RA::Stats::ZeroOut()
{
    SetAllValues(0, false);
}

DXF void RA::Stats::SetAllValues(const double FnValue, const bool FbHadFirstIndent)
{
    MbHadFirstInsert = FbHadFirstIndent;
    MnInsertIdx = 1;
    if (!MvValues)
        return;

    for (double* Ptr = MvValues; Ptr < MvValues + MnStorageSize; Ptr++)
        *Ptr = FnValue * MAX(MoJoinery.MnSize, 1);

    if (MoJoinery.MnSize)
    {
        //for (Cheunk* Ptr = MoJoinery.MvChunks; Ptr < MoJoinery.MvChunks + MnStorageSize; Ptr++)
        //{
        //    auto& Ref = *Ptr;
        //    Ref.MnSum = FnValue * Ref.MnSize;
        //    Ref.MnIdx = 1;
        //    for (auto* Ptr2 = Ref.MvValues; Ptr2 < Ref.MvValues + Ref.MnSize; Ptr2++)
        //        *Ptr2 = FnValue;
        //    Ref.MvValues[0] = FnValue;
        //    Ref.MvValues[Ref.MnSize] = 0;
        //}

        for (auto* Ptr2 = MoJoinery.MvValues; Ptr2 < MoJoinery.MvValues + MoJoinery.MnSize; Ptr2++)
            *Ptr2 = FnValue;
        MoJoinery.MvValues[0] = FnValue;
        MoJoinery.MvValues[MoJoinery.MnSize] = 0;
    }

    SetDefaultValues(FnValue);
}

DXF void RA::Stats::SetDeviceJoinery()
{
    if (MoAvgPtr)
    {
        auto& MoObj = *MoAvgPtr;
        MoObj.MvValues = MvValues;
        MoObj.MnInsertIdxPtr = &MnInsertIdx;
    }
    if (MoRSIPtr)
    {
        auto& MoObj = *MoRSIPtr;
        MoObj.MvValues = MvValues;
        MoObj.MnInsertIdxPtr = &MnInsertIdx;
    }
    if (MoSTOCHPtr)
    {
        auto& MoObj = *MoSTOCHPtr;
        MoObj.MvValues = MvValues;
        MoObj.MnInsertIdxPtr = &MnInsertIdx;
    }
    SRef(MoStandardDeviationPtr).SetAvg(MoAvgPtr);
    SRef(MoMeanAbsoluteDeviationPtr).SetAvg(MoAvgPtr);
}

DXF double RA::Stats::GetSkippedNum(const xint FIdx) const
{
    if (FIdx >= MoSlippage.MnDataLeng)
        printf(RED "Out of Range" WHITE);
    return MoSlippage.MvNums[MoSlippage.MnDataLeng - 1 - FIdx];
}

DXF void RA::Stats::TheJoinery::InsertNum(const double FnNum)
{
    const auto LnReplaced = MvValues[MnIdx];
    MnSum += FnNum;
    MnSum -= LnReplaced;
    MvValues[MnIdx] = FnNum;
    if (MnIdx < MnSize - 1)
        MnIdx++;
    else
        MnIdx = 0;
}
