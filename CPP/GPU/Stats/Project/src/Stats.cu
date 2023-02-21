#if UsingMSVC
#include "Stats.h"
#else
#include "Stats.cuh"
#endif

// Copyright via Apache v2 Licence [2023][Joel Leagues aka Scourge]

RA::Stats::Stats()
{
}

RA::Stats::~Stats()
{
    Clear();
}

void RA::Stats::Clear()
{
    Begin();

    ClearStorageRequriedObjs();
    if (MeHardware == EHardware::GPU) {
        CudaDelete(MvValues);
        CudaDelete(MvJoinery);
    }
    else if (MeHardware == EHardware::CPU) {
        HostDelete(MvValues);
        HostDelete(MvJoinery);
    }
    else
        ThrowIt("EHardware (GPU/CPU) Option Not Given");

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
            break;
        }
        case EHardware::CPU:
        {
            HostDelete(MoAvgPtr);
            HostDelete(MoSTOCHPtr);
            HostDelete(MoRSIPtr);
            break;
        }
        default:
            ThrowIt("EHardware (GPU/CPU) Option Not Given");
    }
    Rescue();
}


void RA::Stats::CreateObjs(const xmap<EOptions, uint>& FmOptions)
{
    Begin();
    ClearStorageRequriedObjs();
    switch (MeHardware)
    {
        case RA::EHardware::CPU:
        {
            for (const std::pair<EOptions, uint>& Pair : FmOptions)
            {
                if (Pair.first == EOptions::AVG)
                    MoAvgPtr = new AVG(MvValues, Pair.second, &MnStorageSize, &MnInsertIdx);
                else if (Pair.first == EOptions::STOCH)
                    MoSTOCHPtr = new STOCH(MvValues, Pair.second, &MnStorageSize, &MnInsertIdx);
                else if (Pair.first == EOptions::RSI)
                    MoRSIPtr = new RSI(MvValues, Pair.second, &MnStorageSize, &MnInsertIdx);
            }
            break;
        }
#ifndef UsingMSVC
        case RA::EHardware::GPU:
        {
            for (const std::pair<EOptions, uint>& Pair : FmOptions)
            {
                if (Pair.first == EOptions::AVG)
                    MoAvgPtr = RA::Host::AllocateObjOnDevice<AVG>(MvValues, Pair.second, &MnStorageSize, &MnInsertIdx);
                else if (Pair.first == EOptions::STOCH)
                    MoSTOCHPtr = RA::Host::AllocateObjOnDevice<STOCH>(MvValues, Pair.second, &MnStorageSize, &MnInsertIdx);
                else if (Pair.first == EOptions::RSI)
                    MoRSIPtr = RA::Host::AllocateObjOnDevice<RSI>(MvValues, Pair.second, &MnStorageSize, &MnInsertIdx);
            }
            break;
        }
#endif
        default:
            ThrowIt("EHardware (GPU/CPU) Option Not Given");
    }
    Rescue();
}

void RA::Stats::Allocate(const uint FnStorageSize, const double FnDefaultVal)
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
            else
                Clear();
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
            else
                Clear();
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
    const uint FnStorageSize,
    const xmap<EOptions, uint>& FmOptions,
    const double FnDefaultVal)
{
    Begin();

    if (!!MmOptions)
    {
        Clear();
        MmOptions.clear();
    }

    MnStorageSize = FnStorageSize;
    MmOptions = FmOptions;

    Allocate(MnStorageSize, FnDefaultVal);
    CreateObjs(FmOptions);

    if (MeHardware == EHardware::CPU)
    {
        SRef(MoAvgPtr).SetDefaultValues(FnDefaultVal);
        SRef(MoSTOCHPtr).SetDefaultValues(FnDefaultVal);
        SRef(MoRSIPtr).SetDefaultValues();
    }

    MnInsertIdx = 0;
    MbHadFirstInsert = false;

    Rescue();
}


void RA::Stats::Construct(
    const EHardware FeHardware,
    const uint FnStorageSize,
    const xmap<EOptions, uint>& FmOptions,
    const double FnDefaultVal)
{
    Begin();
    MeHardware = FeHardware;
    This.Construct(FnStorageSize, FmOptions, FnDefaultVal);
    Rescue();
}

RA::Stats::Stats(
    const EHardware FeHardware,
    const uint FnStorageSize,
    const xmap<EOptions, uint>& FmOptions, // Options <> Logical Size
    const double FnDefaultVal)
{
    Begin();
    This.Construct(FeHardware, FnStorageSize, FmOptions, FnDefaultVal);
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

void RA::Stats::SetDefaultValues(const double FnDefaultVal)
{
    Begin();
    GET(MoAvg);
    GET(MoSTOCH);
    MoAvg.SetDefaultValues(FnDefaultVal);
    MoSTOCH.SetDefaultValues(FnDefaultVal);
    Rescue();
}

void RA::Stats::SetJoinerySize(const uint FCount)
{
    Begin();
    if (FCount <= 1)
        return;
    MnJoinerySize = FCount;
    MnInsertIdx = 0;
    MbHadFirstInsert = false;

    switch (MeHardware)
    {
        case RA::EHardware::CPU:
        {
            if (MnJoinerySize)
            {
                DeleteArr(MvJoinery);
                MvJoinery = new double[MnJoinerySize + 1];
                for (auto* Ptr = MvJoinery; Ptr < MvJoinery + MnJoinerySize; Ptr++)
                    *Ptr = 0;
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
                CudaDelete(MvJoinery);
                auto Ptr = RA::MakeShared<double[]>(MnJoinerySize + 1);
                for (auto& Val : Ptr)
                    Val = FnDefaultVal;
                MvJoinery = RA::Host::AllocateArrOnDevice<double>(Ptr.Raw(), RA::Allocate(MnJoinerySize, sizeof(double)));
            }
            else
                Clear();
            break;
        }
#endif
    default:
        ThrowIt("EHardware (GPU/CPU) Option Not Given");
    }

    Rescue();
}


DXF void RA::Stats::operator<<(const double FnValue)
{
    double LnValue = 0;

    if (MbHadFirstInsert)
    {
        if (!MnJoinerySize)
        {
            MnInsertIdx++;
            if (MnInsertIdx >= MnStorageSize)
                MnInsertIdx = 0;
        }
        else
        {
            if (++MnJoineryIdx >= MnJoinerySize)
                MnJoineryIdx = 0;
            if (!MnJoineryIdx)
            {
                MnInsertIdx++; // you get a new idx when MnSkipIdx are 0
                if (MnInsertIdx >= MnStorageSize)
                    MnInsertIdx = 0;
                //MnJoinerySum = MvValues[MnInsertIdx];

                //const auto LnIdxVal = MnJoinerySum / MnJoinerySize;
                //for (double* Ptr = MvJoinery; Ptr < MvJoinery + MnJoinerySize; Ptr++)
                //    *Ptr = LnIdxVal;
            }
        }
    }

    if (!MnStorageSize)
    {
        if (!MbHadFirstInsert && MnJoinerySize)
        {
            for (double* Ptr = MvJoinery; Ptr < MvJoinery + MnJoinerySize; Ptr++)
                *Ptr = FnValue;
            MnJoinerySum = FnValue * MnJoinerySize;
            MbHadFirstInsert = true;
        }

        if (MnJoinerySize)
        {
            if (++MnJoineryIdx >= MnJoinerySize)
                MnJoineryIdx = 0;

            MnJoinerySum -= MvJoinery[MnJoineryIdx]; // subtract old val
            MvJoinery[MnJoineryIdx] = FnValue; // set new val
            MnJoinerySum += FnValue; // add new val

            LnValue = MnJoinerySum; // set new value to main storage

        }
        else
            LnValue = FnValue;

        SRef(MoAvgPtr).Update(LnValue);
        // storage needed for: Min/Max, RSI, STOCH
        return;
    }

    if (!MbHadFirstInsert)
    {
        double LnFillSize = 0;
        if (MnJoinerySize)
        {
            for (double* Ptr = MvJoinery; Ptr < MvJoinery + MnJoinerySize; Ptr++)
                *Ptr = FnValue;
            MnJoinerySum = FnValue * MnJoinerySize;
            LnFillSize = MnJoinerySum;
        }
        else
            LnFillSize = FnValue;

        for (double* Ptr = MvValues; Ptr < MvValues + MnStorageSize; Ptr++)
            *Ptr = LnFillSize;

        MnInsertIdx = 0;
    }
    else
    {
        if (MnJoinerySize)
        {
            MnJoinerySum -= MvJoinery[MnJoineryIdx]; // subtract old val
            MvJoinery[MnJoineryIdx] = FnValue; // set new val
            MnJoinerySum += FnValue; // add new val

            MvValues[MnInsertIdx] = MnJoinerySum; // set new value to main storage
        }
        else
            MvValues[MnInsertIdx] = FnValue;
    }

    SRef(MoAvgPtr).Update(FnValue);
    SRef(MoRSIPtr).Update();
    SRef(MoSTOCHPtr).Update();

    if (!MbHadFirstInsert)
    {
        MbHadFirstInsert = true;
        return;
    }
}

DXF void RA::Stats::Reset()
{
    SetAllValues(0, false);
}

DXF void RA::Stats::ZeroOut()
{
    SetAllValues(0, false);
}

DXF void RA::Stats::SetAllValues(const double FnValue, const bool FbHadFirstIndent)
{
    MbHadFirstInsert = FbHadFirstIndent;
    if (!MvValues)
        return;
    for (auto Ptr = MvValues; Ptr < MvValues + MnStorageSize; Ptr++)
        *Ptr = FnValue;
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
}


