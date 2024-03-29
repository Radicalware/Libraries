﻿#pragma once
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
            const xvector<EStatOpt> FvOptions,
            const double FnDefaultVal = 0);

        DHF AVG&        GetObjAVG();
        DHF STOCH&      GetObjSTOCH();
        DHF RSI&        GetObjRSI();
        DHF Deviation&  GetObjStandardDeviation();
        DHF Deviation&  GetObjMeanAbsoluteDeviation();

        DHF const AVG&       GetObjAVG()   const;
        DHF const STOCH&     GetObjSTOCH() const;
        DHF const RSI&       GetObjRSI()   const;
        DHF const Deviation& GetObjStandardDeviation() const;
        DHF const Deviation& GetObjMeanAbsoluteDeviation() const;
        
        DHF const AVG&       AVG()   const;
        DHF const STOCH&     STOCH() const;
        DHF const RSI&       RSI()   const;
        DHF const Deviation& SD()    const;
        DHF const Deviation& MAD()   const;

        DHF       Deviation& SD();
        DHF       Deviation& MAD();
        
        IHF double GetAVG()   const { return GetObjAVG().GetAVG(); }
        IHF double GetSTOCH() const { return GetObjSTOCH().GetSTOCH(); }
        IHF double GetRSI()   const { return GetObjRSI().GetRSI(); }
    };
};
