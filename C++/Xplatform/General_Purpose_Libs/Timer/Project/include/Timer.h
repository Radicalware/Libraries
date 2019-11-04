#pragma once
#include<chrono>

#include "xvector.h"
#include "xstring.h"
#include "xmap.h"

class Timer
{
    xvector<double>       m_laps_xv;
    xmap<xstring, double> m_laps_xm;

    using Timer_clock_t = std::chrono::steady_clock;
    using Timer_second_t = std::chrono::duration<double, std::ratio<1> >;
    std::chrono::time_point<Timer_clock_t> m_beg = Timer_clock_t::now();

public:
    Timer();
    void reset();
    double elapsed() const;

    void wait_seconds(double extent) const;
    void wait_milliseconds(unsigned long extent) const;
    void wait(unsigned long extent) const; // wait_milliseconds

    void lap();
    void lap(const xstring& key);
    void clear();

    void sleep(unsigned long extent) const;
    double get(size_t idx) const;
    double get(const xstring& key) const;

    xvector<double> get_xvector() const;
    xmap<xstring, double> get_xmap() const;
};


std::ostream& operator<<(std::ostream& out, const Timer& time);
