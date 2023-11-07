#pragma once

#include<chrono>

#include "xvector.h"
#include "xstring.h"
#include "Macros.h"
#include "xmap.h"

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

namespace RA
{
    class EXI Timer
    {
        xvector<xint>       MvLaps;
        xmap<xstring, xint> MmLaps;

        // SteadyClock = Stopwatch   Use Case
        // SystemClock = Wrist-Watch Use Case
        using SteadyClock = std::chrono::steady_clock;
        std::chrono::time_point<SteadyClock> MoClock = SteadyClock::now();

    public:
        static const Timer StaticClass;

        Timer();
        void Reset();
        xint GetElapsedTimeSeconds() const;
        xint GetElapsedTimeMilliseconds() const;
        xint GetElapsedTime() const; // Milliseconds
        xint GetElapsedTimeMicroseconds() const;

        static void WaitSeconds(xint extent);
        static void WaitMilliseconds(unsigned long extent);
        static void Wait(unsigned long extent); // wait_milliseconds
        static void WaitUntil(unsigned long extent, std::function<bool()>&& Function);
        static void PassOrWait(unsigned long TestEveryTimer, unsigned long ExitAnywayTimer, std::function<bool()>&& Function);
        static void PassOrWaitSeconds(unsigned long TestEveryTimer, unsigned long ExitAnywayTimer, std::function<bool()>&& Function);

        static void Sleep(unsigned long FnMilliseconds);
        static void SleepSeconds(unsigned long FnSeconds);

        void Lap();
        void Lap(const xstring& key);
        void Lap(xstring&& key);
        void Clear();

        xint Get(size_t idx) const;
        xint Get(const xstring& key) const;

        xvector<xint> GetVector() const;
        xmap<xstring, xint> GetMap() const;
    };
};

EXI std::ostream& operator<<(std::ostream& out, const RA::Timer& time);