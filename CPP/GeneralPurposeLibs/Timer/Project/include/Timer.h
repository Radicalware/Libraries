#pragma once

#include<chrono>

#include "xvector.h"
#include "xstring.h"

#ifndef UsingNVCC
#include "xmap.h"
#include "re2/re2.h"
#endif // !UsingNVCC



#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
    #ifdef DLL_EXPORT
       #define EXI __declspec(dllexport)
    #else
       #define EXI __declspec(dllimport)
    #endif
#else
    #define EXI
#endif

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