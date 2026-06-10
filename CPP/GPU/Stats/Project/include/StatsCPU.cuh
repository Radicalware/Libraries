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
            const xvector<EStatOpt> FvOptions,
            const double FnDefaultVal = 0);
        
        DHF const AVG&       AVG()     const { return const_cast<StatsCPU*>(this)->AVG(); }
        DHF const ZEA&       ZEA()     const { return const_cast<StatsCPU*>(this)->ZEA(); }
        DHF const Omaha&     Omaha()   const { return const_cast<StatsCPU*>(this)->Omaha(); }
        DHF const STOCH&     STOCH()   const { return const_cast<StatsCPU*>(this)->STOCH(); }
        DHF const RSI&       RSI()     const { return const_cast<StatsCPU*>(this)->RSI(); }
        DHF const Normals&   Normals() const { return const_cast<StatsCPU*>(this)->Normals(); }
        DHF const Deviation& SD()      const { return const_cast<StatsCPU*>(this)->SD(); }
        DHF const Deviation& MAD()     const { return const_cast<StatsCPU*>(this)->MAD(); }
        
        DHF RA::AVG&         AVG();
        DHF RA::ZEA&         ZEA();
        DHF RA::Omaha&       Omaha();
        DHF RA::STOCH&       STOCH();
        DHF RA::RSI&         RSI();
        DHF RA::Normals&     Normals();
        DHF RA::Deviation&   SD();
        DHF RA::Deviation&   MAD();

        IXF double GetAVG()   const { return The.AVG().GetAVG(); }
        IXF double GetSUM()   const { return The.AVG().GetSum(); }
        IXF double GetZEA()   const { return The.ZEA().GetZEA(); }
        IXF double GetSTOCH() const { return The.STOCH().GetSTOCH(); }
        IXF double GetRSI()   const { return The.RSI().GetRSI(); }
        IXF auto GetNormals() const { return The.Normals().GetNormals(); }
        IXF auto GetNormalFront(const xint Idx = 0) const { return The.Normals().GetNormalFront(Idx); }
        IXF auto GetNormalBack(const xint Idx = 0) const { return The.Normals().GetNormalBack(Idx); }

        IXF double GetScaledSTOCH() const { return The.STOCH().GetScaledSTOCH(); }
        IXF double GetScaledRSI()   const { return The.RSI().GetScaledRSI(); }

        DHF double Get(const RA::EStatOpt) const;
    };
};
