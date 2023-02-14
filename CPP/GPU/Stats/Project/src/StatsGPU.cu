// Copyright via Apache v2 Licence [2023][Joel Leagues aka Scourge]
#if UsingMSVC
#include "StatsGPU.h"
#else
#include "StatsGPU.cuh"
#endif

#ifndef UsingMSVC

#include "Memory.h"
#include "Host.cuh"

#include "CudaBridge.cuh"

RA::StatsGPU::StatsGPU()
{
    MeHardware = EHardware::GPU;
}

RA::StatsGPU::StatsGPU(const StatsGPU& Other)
{
    Begin();
    This = Other;
    Rescue();
}

RA::StatsGPU::StatsGPU(StatsGPU&& Other) noexcept
{
    This = std::move(Other);
}

void RA::StatsGPU::operator=(const StatsGPU& Other)
{
    Begin();
    MeHardware = EHardware::GPU;
    if (Other.MvValues && Other.MnStorageSize)
    {
        Allocate(Other.MnStorageSize, 1);
        cudaMemcpy(MvValues, Other.MvValues, Other.MnStorageSize, cudaMemcpyDeviceToDevice);
        MbHadFirstInsert = Other.MbHadFirstInsert;
        MnInsertIdx = Other.MnInsertIdx;
    }
    else
    {
        MnInsertIdx = 0;
        MbHadFirstInsert = false;
    }

    if (Other.MoAvgPtr)
        cudaMemcpy(MoAvgPtr, Other.MoAvgPtr, sizeof(RA::AVG), cudaMemcpyDeviceToDevice);

    if (Other.MoSTOCHPtr)
        cudaMemcpy(MoSTOCHPtr, Other.MoSTOCHPtr, sizeof(RA::STOCH), cudaMemcpyDeviceToDevice);

    Rescue();
}

void RA::StatsGPU::operator=(StatsGPU&& Other) noexcept
{
    CudaDelete(MvValues);
    if (Other.MvValues && Other.MnStorageSize)
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

RA::StatsGPU::StatsGPU(
    const uint FnStorageSize,
    const xmap<EOptions, uint>& FmOptions,
    const double FnDefaultVal) :
    RA::Stats(RA::EHardware::GPU, FnStorageSize, FmOptions, FnDefaultVal)
{
}


DDF double RA::StatsGPU::operator[](const uint IDX) const
{
    if (IDX >= MnStorageSize)
        printf(RED "IDX = %llu which is too big for size of\n" WHITE, MnStorageSize);
    return MvValues[IDX];
}

DDF double RA::StatsGPU::Last(const uint IDX) const
{
    if (IDX >= MnStorageSize)
        printf(RED "IDX = %llu which is too big for size of\n" WHITE, MnStorageSize);
    return MvValues[MnStorageSize - 1 - IDX];
}

// --------------------------------------------------------

DDF RA::AVG& RA::StatsGPU::GetObjAVG()
{
    if (!MoAvgPtr)
    {
        printf(RED "MoAvgPtr is Null" WHITE);
        return *MoAvgPtr;
    }
    return *MoAvgPtr;
}

DDF RA::STOCH& RA::StatsGPU::GetObjSTOCH()
{
    if (!MoSTOCHPtr)
    {
        printf(RED "MoSTOCHPtr is Null" WHITE);
        return *MoSTOCHPtr;
    }
    return *MoSTOCHPtr;
}

DDF RA::RSI& RA::StatsGPU::GetObjRSI()
{
    if (!MoRSIPtr)
    {
        printf(RED "MoRSIPtr is Null" WHITE);
        return *MoRSIPtr;
    }
    return *MoRSIPtr;
}

// --------------------------------------------------------

DDF const RA::AVG& RA::StatsGPU::GetObjAVG() const
{
    if (!MoAvgPtr)
    {
        printf(RED "MoAvgPtr is Null" WHITE);
        return *MoAvgPtr;
    }
    return *MoAvgPtr;
}

DDF const RA::STOCH& RA::StatsGPU::GetObjSTOCH() const
{
    if (!MoSTOCHPtr)
    {
        printf(RED "MoSTOCHPtr is Null" WHITE);
        return *MoSTOCHPtr;
    }
    return *MoSTOCHPtr;
}

DDF const RA::RSI& RA::StatsGPU::GetObjRSI() const
{
    if (!MoRSIPtr)
    {
        printf(RED "MoRSIPtr is Null" WHITE);
        return *MoRSIPtr;
    }
    return *MoRSIPtr;
}

// --------------------------------------------------------

DDF const RA::AVG& RA::StatsGPU::AVG() const
{
    if (!MoAvgPtr)
    {
        printf(RED "MoAvgPtr is Null" WHITE);
        return *MoAvgPtr;
    }
    return *MoAvgPtr;
}

DDF const RA::STOCH& RA::StatsGPU::STOCH() const
{
    if (!MoSTOCHPtr)
    {
        printf(RED "MoSTOCHPtr is Null" WHITE);
        return *MoSTOCHPtr;
    }
    return *MoSTOCHPtr;
}

DDF const RA::RSI& RA::StatsGPU::RSI() const
{
    if (!MoRSIPtr)
    {
        printf(RED "MoRSIPtr is Null" WHITE);
        return *MoRSIPtr;
    }
    return *MoRSIPtr;
}
// --------------------------------------------------------

__global__ void RA::Device::ConfigureStats(RA::StatsGPU* StatsPtr)
{
    auto& Stats = *StatsPtr;
    Stats.SetDeviceJoinery();
}

void RA::Host::ConfigureStats(RA::StatsGPU* StatsPtr)
{
    RA::CudaBridge<>::NONE::RunGPU(dim3(1), dim3(1), &Device::ConfigureStats, StatsPtr);
}
void RA::Host::ConfigureStatsSync(RA::StatsGPU* StatsPtr)
{
    RA::CudaBridge<>::NONE::RunGPU(dim3(1), dim3(1), &Device::ConfigureStats, StatsPtr);
    RA::CudaBridge<>::SyncAll();
}

#endif