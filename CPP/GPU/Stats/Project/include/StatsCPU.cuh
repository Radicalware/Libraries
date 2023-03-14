#pragma once
// Copyright via Apache v2 Licence [2023][Joel Leagues aka Scourge]

#if UsingMSVC
#include "Stats.h"
#else
#include "Stats.cuh"
#endif

#include "Macros.h"

namespace RA
{
    class StatsCPU : public Stats
    {
    public:
        using Stats::Stats;
        StatsCPU();
        StatsCPU(const StatsCPU& Other);
        StatsCPU(StatsCPU&& Other) noexcept;
        void operator=(const StatsCPU& Other);
        void operator=(StatsCPU&& Other) noexcept;
        StatsCPU(
            const xint FnStorageSize,
            const xmap<EOptions, xint>& FmOptions, // Options <> Logical Size
            const double FnDefaultVal = 0);

        DHF AVG&   GetObjAVG();
        DHF STOCH& GetObjSTOCH();
        DHF RSI&   GetObjRSI();

        DHF const AVG&   GetObjAVG()   const;
        DHF const STOCH& GetObjSTOCH() const;
        DHF const RSI&   GetObjRSI()   const;

        DHF const AVG&   AVG()   const;
        DHF const STOCH& STOCH() const;
        DHF const RSI&   RSI()   const;
        
        IHF double GetAVG()   const { return GetObjAVG().GetAVG(); }
        IHF double GetSTOCH() const { return GetObjSTOCH().GetSTOCH(); }
        IHF double GetRSI()   const { return GetObjRSI().GetRSI(); }
    };
};
