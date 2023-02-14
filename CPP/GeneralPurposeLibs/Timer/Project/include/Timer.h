#pragma once

#include<chrono>

#include "xvector.h"
#include "xstring.h"
#include "Macros.h"

#ifndef UsingNVCC
#include "xmap.h"
#include "re2/re2.h"
#endif // !UsingNVCC

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

#ifndef UsingNVCC
        xvector<uint>       m_laps_xv;
        xmap<xstring, uint> m_laps_xm;
#endif // !UsingNVCC

        // SteadyClock = Stopwatch   Use Case
        // SystemClock = Wrist-Watch Use Case
        using SteadyClock = std::chrono::steady_clock;
        std::chrono::time_point<SteadyClock> m_beg = SteadyClock::now();

    public:
        static const Timer StaticClass;

        Timer();
        void Reset();
        uint GetElapsedTimeSeconds() const;
        uint GetElapsedTimeMilliseconds() const;
        uint GetElapsedTime() const; // Milliseconds
        uint GetElapsedTimeMicroseconds() const;

        static void WaitSeconds(uint extent);
        static void WaitMilliseconds(unsigned long extent);
        static void Wait(unsigned long extent); // wait_milliseconds
        static void WaitUntil(unsigned long extent, std::function<bool()>&& Function);
        static void PassOrWait(unsigned long TestEveryTimer, unsigned long ExitAnywayTimer, std::function<bool()>&& Function);
        static void PassOrWaitSeconds(unsigned long TestEveryTimer, unsigned long ExitAnywayTimer, std::function<bool()>&& Function);

        static void Sleep(unsigned long FnMilliseconds);
        static void SleepSeconds(unsigned long FnSeconds);

#ifndef UsingNVCC
        void Lap();
        void Lap(const xstring& key);
        void Lap(xstring&& key);
        void Clear();

        uint Get(size_t idx) const;
        uint Get(const xstring& key) const;

        xvector<uint> GetVector() const;
        xmap<xstring, uint> GetMap() const;
#endif // !UsingNVCC
    };
};

EXI std::ostream& operator<<(std::ostream& out, const RA::Timer& time);