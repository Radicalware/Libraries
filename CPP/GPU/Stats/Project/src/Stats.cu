#if UsingMSVC
#include "Stats.h"
#else
#include "Stats.cuh"
#endif
#include "Memory.h"

// Copyright via Apache v2 Licence [2023][Joel Leagues aka Scourge]

DXF double RA::Joinery::InsertNum(const double FnNum)
{
    const auto LnReplaced = MvValues[MnIdx];
    MnSum += FnNum;
    MnSum -= LnReplaced;
    MvValues[MnIdx] = FnNum;
    if (MnIdx < MnSize - 1)
        MnIdx++;
    else
        MnIdx = 0;
    return LnReplaced;
}

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

    if (MeHardware == EHardware::GPU)
    {
        if (MvJoinery)
        {
            for (xint i = 0; i < MnStorageSize; i++) {
                CudaDelete(MvJoinery[i].MvValues);
            }
            CudaDelete(MvJoinery);
        }
    }
    else if (MeHardware == EHardware::CPU)
    {
        if (MvJoinery)
        {
            for (xint i = 0; i < MnStorageSize; i++) {
                HostDelete(MvJoinery[i].MvValues);
            }
            HostDelete(MvJoinery);
        }
    }
    else
        ThrowIt("EHardware (GPU/CPU) Option Not Given");
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


void RA::Stats::CreateObjs(const xmap<EStatOpt, xint>& FmOptions)
{
    Begin();
    MmOptions = FmOptions;
    ClearStorageRequriedObjs();
    switch (MeHardware)
    {
    case RA::EHardware::CPU:
    {
        for (const std::pair<EStatOpt, xint>& Pair : FmOptions)
        {
            if (Pair.first == EStatOpt::AVG)
                MoAvgPtr = new AVG(MvValues, Pair.second, &MnStorageSize, &MnInsertIdx);
            else if (Pair.first == EStatOpt::STOCH)
                MoSTOCHPtr = new STOCH(MvValues, Pair.second, &MnStorageSize, &MnInsertIdx);
            else if (Pair.first == EStatOpt::RSI)
                MoRSIPtr = new RSI(MvValues, Pair.second, &MnStorageSize, &MnInsertIdx);
            else if (Pair.first == EStatOpt::SD)
                MoStandardDeviationPtr = new Deviation(MoAvgPtr, Deviation::EType::SD);
            else if (Pair.first == EStatOpt::MAD)
                MoMeanAbsoluteDeviationPtr = new Deviation(MoAvgPtr, Deviation::EType::MAD);
        }
        break;
    }
#ifndef UsingMSVC
    case RA::EHardware::GPU:
    {
        for (const std::pair<EStatOpt, xint>& Pair : FmOptions)
        {
            if (Pair.first == EStatOpt::AVG)
                MoAvgPtr = RA::Host::AllocateObjOnDevice<AVG>(MvValues, Pair.second, &MnStorageSize, &MnInsertIdx);
            else if (Pair.first == EStatOpt::STOCH)
                MoSTOCHPtr = RA::Host::AllocateObjOnDevice<STOCH>(MvValues, Pair.second, &MnStorageSize, &MnInsertIdx);
            else if (Pair.first == EStatOpt::RSI)
                MoRSIPtr = RA::Host::AllocateObjOnDevice<RSI>(MvValues, Pair.second, &MnStorageSize, &MnInsertIdx);
            else if (Pair.first == EStatOpt::SD)
                MoStandardDeviationPtr = RA::Host::AllocateObjOnDevice<Deviation>(MoAvgPtr, Deviation::EType::SD);
            else if (Pair.first == EStatOpt::MAD)
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
    const xmap<EStatOpt, xint>& FmOptions,
    const double FnDefaultVal)
{
    Begin();

    if (!!MmOptions)
    {
        Clear();
        MmOptions.clear();
    }

    MnStorageSize = FnStorageSize;

    Allocate(MnStorageSize, FnDefaultVal);
    CreateObjs(FmOptions);

    if (MeHardware == EHardware::CPU && FnDefaultVal)
        SetDefaultValues(FnDefaultVal);

    MnInsertIdx = 0;
    MbHadFirstInsert = false;

    Rescue();
}


void RA::Stats::ConstructHardware(
    const EHardware FeHardware,
    const xint FnStorageSize,
    const xmap<EStatOpt, xint>& FmOptions,
    const double FnDefaultVal)
{
    Begin();
    MeHardware = FeHardware;
    The.Construct(FnStorageSize, FmOptions, FnDefaultVal);
    Rescue();
}

RA::Stats::Stats(
    const EHardware FeHardware,
    const xint FnStorageSize,
    const xmap<EStatOpt, xint>& FmOptions, // Options <> Logical Size
    const double FnDefaultVal)
{
    Begin();
    The.ConstructHardware(FeHardware, FnStorageSize, FmOptions, FnDefaultVal);
    Rescue();
}


void RA::Stats::SetStorageSizeZero(const double FnDefaultVal)
{
    Begin();
    if (MvValues)
    {
#ifdef UsingNVCC
        CudaDelete(MvValues);
#else
        HostDelete(MvValues);
#endif
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
    MnJoinerySize = FCount;
    MnInsertIdx = 0;
    MbHadFirstInsert = false;
    ClearJoinery();

    switch (MeHardware)
    {
    case RA::EHardware::CPU:
    {
        if (MnJoinerySize)
        {
            MvJoinery = new Joinery[MnStorageSize + 1];
            for (auto* Ptr = MvJoinery; Ptr < MvJoinery + MnStorageSize; Ptr++)
            {
                auto& Ref = *Ptr;
                Ref.MnSum = 0;
                Ref.MnIdx = 0;
                Ref.MnSize = MnJoinerySize;
                Ref.MvValues = new double[Ref.MnSize + 1];
                for (auto* Ptr2 = Ref.MvValues; Ptr2 <= Ref.MvValues + Ref.MnSize; Ptr2++)
                    *Ptr2 = 0;
            }
        }
        else
            Clear();
        break;
    }
#ifndef UsingMSVC
    case RA::EHardware::GPU:
    {
        if (MnJoinerySize)
        {
            auto Ptr = RA::MakeShared<Joinery[]>(MnStorageSize + 1);
            for (auto& Val : Ptr)
            {
                Val.MnSum = 0;
                Val.MnIdx = 0;
                Val.MnSize = MnJoinerySize;
                Val.MvValues = new double[Val.MnSize + 1];
                for (auto* Ptr2 = Val.MvValues; Ptr2 <= Val.MvValues + Val.MnSize; Ptr2++)
                    *Ptr2 = 0;
            }
            MvJoinery = RA::Host::AllocateArrOnDevice<Joinery>(
                Ptr.Raw(),
                RA::Allocate(MnJoinerySize, sizeof(Joinery) + (sizeof(double) * MnJoinerySize + sizeof(double))));
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


DXF void RA::Stats::SetMaxTraceSize(const xint FSize)
{
    MnMaxTraceSize = FSize;
    SRef(MoAvgPtr).SetMaxTraceSize(FSize);
    SRef(MoStandardDeviationPtr).SetMaxTraceSize(FSize);
    SRef(MoMeanAbsoluteDeviationPtr).SetMaxTraceSize(FSize);
}

DXF double RA::Stats::operator[](const xint IDX) const
{
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
    if (!MnStorageSize || !MvValues)
    {
        MnLastValue = FnValue;
        SRef(MoAvgPtr).Update(FnValue);
        SRef(MoStandardDeviationPtr).Update(FnValue);
        SRef(MoMeanAbsoluteDeviationPtr).Update(FnValue);
        SRef(MoSTOCHPtr).Update(FnValue);
        return;
    }

    if (MbHadFirstInsert)
    {
        if (!MnJoinerySize)
        {
            MnInsertIdx++;
            if (MnInsertIdx >= MnStorageSize)
                MnInsertIdx = 0;
        }
    }

    //if (!MbHadFirstInsert)
    //{
    //    SetAllValues(FnValue, true);
    //    MnInsertIdx = (MnStorageSize > 1) ? 1 : 0;
    //    return;
    //}


    // You must send new val as an argument so you
    // have the old and new value instead of just
    // the new value
    SRef(MoAvgPtr).Update(FnValue);
    SRef(MoStandardDeviationPtr).Update(FnValue);

    // SRef(MoMeanAbsoluteDeviationPtr).Update(FnValue);
    if (MoMeanAbsoluteDeviationPtr)
        (*MoMeanAbsoluteDeviationPtr).Update(FnValue);


    if (MnJoinerySize)
    {
        auto LnVal = FnValue;
        auto IDX = MnInsertIdx;
        for (xint i = 0; i < MnStorageSize; i++)
        {
            LnVal = MvJoinery[IDX].InsertNum(LnVal);
            MvValues[IDX] = MvJoinery[IDX].MnSum;
            IDX = (IDX == 0) ? MnStorageSize - 1 : IDX - 1;
        }

    }
    else
        MvValues[MnInsertIdx] = FnValue;

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
        *Ptr = FnValue * MAX(MnJoinerySize, 1);

    if (MnJoinerySize)
    {
        for (Joinery* Ptr = MvJoinery; Ptr < MvJoinery + MnStorageSize; Ptr++)
        {
            auto& Ref = *Ptr;
            Ref.MnSum = FnValue * Ref.MnSize;
            Ref.MnIdx = 1;
            for (auto* Ptr2 = Ref.MvValues; Ptr2 < Ref.MvValues + Ref.MnSize; Ptr2++)
                *Ptr2 = FnValue;
            Ref.MvValues[0] = FnValue;
            Ref.MvValues[Ref.MnSize] = 0;
        }
    }

    SetDefaultValues(FnValue);
}

DXF void RA::Stats::SetDeviceJoinery()
{
    if (MoAvgPtr)
    {
        auto& MoObj = *MoAvgPtr;
        MoObj.MvValues = MvValues;
        MoObj.MnStorageSizePtr = &MnStorageSize;
        MoObj.MnInsertIdxPtr = &MnInsertIdx;
    }
    if (MoRSIPtr)
    {
        auto& MoObj = *MoRSIPtr;
        MoObj.MvValues = MvValues;
        MoObj.MnStorageSizePtr = &MnStorageSize;
        MoObj.MnInsertIdxPtr = &MnInsertIdx;
    }
    if (MoSTOCHPtr)
    {
        auto& MoObj = *MoSTOCHPtr;
        MoObj.MvValues = MvValues;
        MoObj.MnStorageSizePtr = &MnStorageSize;
        MoObj.MnInsertIdxPtr = &MnInsertIdx;
    }
    SRef(MoStandardDeviationPtr).SetAvg(MoAvgPtr);
    SRef(MoMeanAbsoluteDeviationPtr).SetAvg(MoAvgPtr);
}
