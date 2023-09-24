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
            const xint FnStorageSize,
            const xmap<EStatOpt, xint>& FmOptions, // Options <> Logical Size
            const double FnDefaultVal = 0);

        DDF AVG&        GetObjAVG();
        DDF STOCH&      GetObjSTOCH();
        DDF RSI&        GetObjRSI();
        DDF Deviation&  GetObjStandardDeviation();
        DDF Deviation&  GetObjMeanAbsoluteDeviation();

        DDF const AVG&       GetObjAVG()   const;
        DDF const STOCH&     GetObjSTOCH() const;
        DDF const RSI&       GetObjRSI()   const;
        DDF const Deviation& GetObjStandardDeviation() const;
        DDF const Deviation& GetObjMeanAbsoluteDeviation() const;
        
        DDF const AVG&       AVG()   const;
        DDF const STOCH&     STOCH() const;
        DDF const RSI&       RSI()   const;
        DDF const Deviation& SD()    const;
        DDF const Deviation& MAD()   const;

        DDF       Deviation& SD();
        DDF       Deviation& MAD();
        
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
