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
#include "AVG.h"
#include "STOCH.h"
#include "RSI.h"
#else
#include "AVG.cuh"
#include "STOCH.cuh"
#include "RSI.cuh"
#endif

namespace RA
{
    struct Joinery
    {
        double MnSum = 0;
        xint MnIdx = 0;
        xint MnSize = 0;
        double* MvValues = nullptr;

        DXF double InsertNum(const double FnNum);
    };

    class Stats
    {        
    public:
        enum class EOptions : int
        {
            NONE,
            AVG,
            RSI,
            STOCH
        };

    protected:
        Stats(); // use StatsGPU or StatsCPU
        ~Stats();
        void Clear();
        void ClearJoinery();
        void ClearStorageRequriedObjs();

        void CreateObjs(const xmap<EOptions, xint>& FmOptions);
        void Allocate(const xint FnStorageSize, const double FnDefaultVal = 0);

    public:
        void Construct(
            const EHardware FeHardware,
            const xint FnStorageSize,
            const xmap<EOptions, xint>& FmOptions, // Options <> Logical Size
            const double FnDefaultVal = 0);

        void Construct(
            const xint FnStorageSize,
            const xmap<EOptions, xint>& FmOptions, // Options <> Logical Size
            const double FnDefaultVal = 0);
    protected:
        Stats(
            const EHardware FeHardware,
            const xint FnStorageSize,
            const xmap<EOptions, xint>& FmOptions, // Options <> Logical Size
            const double FnDefaultVal = 0);

        void SetStorageSizeZero(const double FnDefaultVal);
        void SetDefaultValues(const double FnDefaultVal);


    public:
        void SetJoinerySize(const xint FCount); // Sums FCount values as one value
        DXF auto GetJoinerySize() const { return MnJoinerySize; }

        DXF double  operator[](const xint IDX) const;
        DXF double  Recent(const xint IDX = 0) const { return The[IDX]; }
        DXF double  Former(const xint IDX = 0) const;

        DXF void operator<<(const double FnValue);

        DXF void Reset();
        DXF void ZeroOut();
        DXF void SetAllValues(const double FnValue, const bool FbHadFirstIndent);

        DXF auto GetStorageSize()  const { return MnStorageSize; }
        DXF auto GetInsertIdx()    const { return MnInsertIdx; }
        DXF auto GetCurrentValue() const { return MvValues[MnInsertIdx]; }
        DXF void SetDeviceJoinery();

        DXF bool BxTrackingAvg()   const { return MoAvgPtr   != nullptr; }
        DXF bool BxTrackingRSI()   const { return MoRSIPtr   != nullptr; }
        DXF bool BxTrackingSTOCH() const { return MoSTOCHPtr != nullptr; }
        DXF void SetHadFirstInsert(const bool FbTruth) { MbHadFirstInsert = FbTruth; }

    protected:
        double*  MvValues = nullptr;
        xint     MnStorageSize = 0;
        xint     MnInsertIdx = 0;

        Joinery* MvJoinery = nullptr;
        xint     MnJoinerySize = 0; // for grouping input values as one combined value every Size times

        bool     MbDelete = true;
        bool     MbHadFirstInsert = false;

        xmap<EOptions, xint> MmOptions;

        AVG   *MoAvgPtr   = nullptr;
        STOCH *MoSTOCHPtr = nullptr;
        RSI   *MoRSIPtr   = nullptr;
        //STOCH     *MoSTOCHPtr = nullptr;

        EHardware MeHardware = EHardware::Default;
    };
}

