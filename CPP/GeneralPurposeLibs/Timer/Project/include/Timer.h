#pragma once

#include<chrono>

#include "xvector.h"
#include "xstring.h"
#include "re2/re2.h"
#include "xmap.h"


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
        xvector<pint>       m_laps_xv;
        xmap<xstring, pint> m_laps_xm;

        // SteadyClock = Stopwatch   Use Case
        // SystemClock = Wrist-Watch Use Case

        using SteadyClock = std::chrono::steady_clock;
        std::chrono::time_point<SteadyClock> m_beg = SteadyClock::now();

    public:
        Timer();
        void Reset();
        INL pint GetElapsedTimeSeconds() const;
        INL pint GetElapsedTimeMilliseconds() const;
        INL pint GetElapsedTime() const; // Milliseconds
        INL pint GetElapsedTimeMicroseconds() const;

        void WaitSeconds(pint extent) const;
        void WaitMilliseconds(unsigned long extent) const;
        void Wait(unsigned long extent) const; // wait_milliseconds

        void Lap();
        void Lap(const xstring& key);
        void Lap(xstring&& key);
        void Clear();

        pint Get(size_t idx) const;
        pint Get(const xstring& key) const;

        xvector<pint> GetVector() const;
        xmap<xstring, pint> GetMap() const;

        static void Sleep(unsigned long FnMilliseconds);
    };
};

EXI std::ostream& operator<<(std::ostream& out, const RA::Timer& time);

INL pint RA::Timer::GetElapsedTimeSeconds() const {
    return static_cast<pint>(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - m_beg).count());
}

INL pint RA::Timer::GetElapsedTimeMilliseconds() const {
    return static_cast<pint>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - m_beg).count());
}

INL pint RA::Timer::GetElapsedTime() const {
    return static_cast<pint>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - m_beg).count());
}

INL pint RA::Timer::GetElapsedTimeMicroseconds() const {
    return static_cast<pint>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - m_beg).count());
}