#pragma once

/*
*|| Copyright[2023][Joel Leagues aka Scourge]
*|| https://GitHub.com/Radicalware
*|| Scourge /at\ protonmail /dot\ com
*|| www.Radicalware.net
*|| https://www.youtube.com/channel/UCivwmYxoOdDT3GmDnD0CfQA/playlists   
*||
*|| Licensed under the Apache License, Version 2.0 (the "License");
*|| you may not use this file except in compliance with the License.        
*|| You may obtain a copy of the License at
*||
*|| http ://www.apache.org/licenses/LICENSE-2.0
*||
*|| Unless required by applicable law or agreed to in writing, software     
*|| distributed under the License is distributed on an "AS IS" BASIS,       
*|| WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*|| See the License for the specific language governing permissions and     
*|| limitations under the License.
*/

#ifndef UsingMSVC
#include "ImportCUDA.cuh"
#endif

#include "Macros.h"
#include "xmap.h"

#if UsingMSVC
#include "Support.h"
#include "AVG.h"
#include "STOCH.h"
#include "RSI.h"
#include "Deviation.h"
#else
#include "Support.cuh"
#include "AVG.cuh"
#include "STOCH.cuh"
#include "RSI.cuh"
#include "Deviation.cuh"
#endif

namespace RA
{
    class Stats
    {        
    protected:
        Stats(); // use StatsGPU or StatsCPU
        void Clear();
        void ClearJoinery();
        void ClearSlippage();
        void ClearStorageRequriedObjs();

        void CreateObjs(const xvector<EStatOpt>& FvOptions);
        void Allocate(const xint FnStorageSize, const double FnDefaultVal = 0);

    public:
        ~Stats();
        void ConstructHardware(
            const EHardware FeHardware,
            const xint FnStorageSize,
            const xvector<EStatOpt>& FvOptions,
            const double FnDefaultVal = 0);

        void Construct(
            const xint FnStorageSize,
            const xvector<EStatOpt>& FvOptions,
            const double FnDefaultVal = 0);
    protected:
        Stats(
            const EHardware FeHardware,
            const xint FnStorageSize,
            const xvector<EStatOpt>& FvOptions,
            const double FnDefaultVal = 0);

        void SetStorageSizeZero(const double FnDefaultVal);
        DXF void SetDefaultValues(const double FnDefaultVal);

    public:
        void SetJoinerySize(const xint FCount); // Sums FCount values as one value
        void SetSlippageSize(const xint FCount);
        DXF auto GetJoinerySize() const { return MoJoinery.MnSize; }
        
        DXF void SetMaxTraceSize(const xint FSize);
        DXF auto GetMaxTraceSize() const { return MnMaxTraceSize; }

        DXF double  operator[](const xint IDX) const;
        DXF double  Recent(const xint IDX = 0) const { return The[IDX]; }
        DXF double  Former(const xint IDX = 0) const;

        DXF void operator<<(const double FnValue);

        DXF void Reset();
        DXF void ZeroOut();
        DXF void SetAllValues(const double FnValue, const bool FbHadFirstIndent);

        DXF auto GetStorageSize()  const { return MnStorageSize; }
        DXF auto GetInsertIdx()    const { return MnInsertIdx; }
        DXF auto GetCurrentValue() const { return (MnStorageSize) ? MvValues[MnInsertIdx] : MnValue; }
        DXF void SetDeviceJoinery();

        DXF bool BxTrackingAvg()   const { return MoAvgPtr   != nullptr; }
        DXF bool BxTrackingRSI()   const { return MoRSIPtr   != nullptr; }
        DXF bool BxTrackingSTOCH() const { return MoSTOCHPtr != nullptr; }
        DXF bool BxTrackingDeviation() const { return MoSTOCHPtr != nullptr; }
        DXF void SetHadFirstInsert(const bool FbTruth) { MbHadFirstInsert = FbTruth; }

        DXF double GetSkippedNum(const xint FIdx) const;
        DXF auto   GetSkipDataLeng() const { return MoSlippage.MnDataLeng; }

    protected:
        struct TheJoinery
        {
            //Chunk* MvChunks = nullptr;
            xint MnIdx = 0;
            xint MnSize = 0;
            double MnSum = 0;
            double* MvValues = nullptr;
            DXF void InsertNum(const double FnNum);
        };

        struct TheSlippage
        {
            xint MnDataLeng = 0;
            xint MnSlipSize = 0;
            EHardware MeHardware = EHardware::Default;
            double* MvNums = nullptr;
        };

        EHardware MeHardware = EHardware::Default;
        xint     MnStorageSize = 0;
        xvector<EStatOpt> MvOptions;

        double   MnValue = 0;
        double*  MvValues = nullptr;
        xint     MnInsertIdx = 0;

        TheJoinery  MoJoinery;
        TheSlippage MoSlippage;

        xint     MnMaxTraceSize = 0; // 0 = infinite; any other is the max divisor for avgs

        bool     MbDelete = true;
        bool     MbHadFirstInsert = false;


        AVG         *MoAvgPtr   = nullptr;
        STOCH       *MoSTOCHPtr = nullptr;
        RSI         *MoRSIPtr   = nullptr;
        Deviation   *MoMeanAbsoluteDeviationPtr = nullptr;
        Deviation   *MoStandardDeviationPtr     = nullptr;

    };
}
