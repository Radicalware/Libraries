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


class EXI Timer
{
    xvector<double>       m_laps_xv;
    xmap<xstring, double> m_laps_xm;

    using Timer_clock_t = std::chrono::steady_clock;
    using Timer_second_t = std::chrono::duration<double, std::ratio<1> >;
    std::chrono::time_point<Timer_clock_t> m_beg = Timer_clock_t::now();

public:
    Timer();
    void Reset();
    double GetElapsedTime() const;

    void WaitSeconds(double extent) const;
    void WaitMilliseconds(unsigned long extent) const;
    void Wait(unsigned long extent) const; // wait_milliseconds

    void Lap();
    void Lap(const xstring& key);
    void Lap(xstring&& key);
    void Clear();

    double Get(size_t idx) const;
    double Get(const xstring& key) const;

    xvector<double> GetVector() const;
    xmap<xstring, double> GetMap() const;

    static void Sleep(unsigned long FnMilliseconds);
};

EXI std::ostream& operator<<(std::ostream& out, const Timer& time);
