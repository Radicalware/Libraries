#pragma once
// Copyright via Apache v2 Licence [2023][Joel Leagues aka Scourge]

#ifndef UsingMSVC

#include "Stats.cuh"
#include "Macros.h"

namespace RA
{
    class StatsGPU : public Stats
    {
    public:
        using Stats::Stats;
        StatsGPU();
        StatsGPU(const StatsGPU& Other);
        StatsGPU(StatsGPU&& Other) noexcept;
        void operator=(const StatsGPU& Other);
        void operator=(StatsGPU&& Other) noexcept;
        StatsGPU(
            const uint FnStorageSize,
            const xmap<EOptions, uint>& FmOptions, // Options <> Logical Size
            const double FnDefaultVal = 0);

        DDF double  operator[](const uint IDX) const;
        //DDF double& operator[](const uint IDX) = delete;
        DDF double  Last(const uint IDX = 0) const;
        //DDF double& Last(const uint IDX = 0) = delete;

        DDF AVG&   GetObjAVG();
        DDF STOCH& GetObjSTOCH();
        DDF RSI&   GetObjRSI();

        DDF const AVG&   GetObjAVG()   const;
        DDF const STOCH& GetObjSTOCH() const;
        DDF const RSI&   GetObjRSI()   const;

        DDF const AVG&   AVG()   const;
        DDF const STOCH& STOCH() const;
        DDF const RSI&   RSI()   const;

        IDF double GetAVG()   const { return GetObjAVG().GetAVG(); }
        IDF double GetSTOCH() const { return GetObjSTOCH().GetSTOCH(); }
        IDF double GetRSI()   const { return GetObjRSI().GetRSI(); }
    };
};


namespace RA
{
    namespace Device
    {
        __global__ void ConfigureStats(RA::StatsGPU* StatsPtr);
    }
};

namespace RA
{
    namespace Host
    {
        void ConfigureStats(RA::StatsGPU* StatsPtr);
        void ConfigureStatsSync(RA::StatsGPU* StatsPtr);
    }
}

#endif