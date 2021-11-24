
#include "Timer.h"
#include "xstring.h"

#include <chrono>

RA::Timer::Timer() : m_beg(SteadyClock::now()) {   }

void RA::Timer::Reset() {
    m_beg = SteadyClock::now();
}


void RA::Timer::WaitSeconds(pint extent) const
{
    while (This.GetElapsedTimeMilliseconds() < extent)
        RA::Timer::Sleep(1);
}

void RA::Timer::WaitMilliseconds(unsigned long extent) const
{
    while (This.GetElapsedTimeMilliseconds() < extent / static_cast<pint>(1000))
        RA::Timer::Sleep(1);
}

void RA::Timer::Wait(unsigned long extent) const
{
    This.WaitMilliseconds(extent);
}

void RA::Timer::Lap()
{
    m_laps_xv << This.GetElapsedTimeMilliseconds();
}

void RA::Timer::Lap(const xstring& key)
{
    pint val = This.GetElapsedTimeMilliseconds();
    m_laps_xm.AddPair(key, val);
    m_laps_xv.push_back(val);
}

void RA::Timer::Lap(xstring&& key)
{
    Lap(key);
}

void RA::Timer::Clear()
{
    m_laps_xv.clear();
    m_laps_xm.clear();
}

pint RA::Timer::Get(size_t idx) const
{
    return m_laps_xv[idx];
}

pint RA::Timer::Get(const xstring& key) const
{
    return m_laps_xm.at(key);
}

xvector<pint> RA::Timer::GetVector() const
{
    return m_laps_xv;
}

xmap<xstring, pint> RA::Timer::GetMap() const
{
    return m_laps_xm;
}

std::ostream& operator<<(std::ostream& out, const RA::Timer& time)
{
    out << time.GetElapsedTimeMilliseconds();
    return out;
}

void RA::Timer::Sleep(unsigned long FnMilliseconds)
{
#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
    ::Sleep(FnMilliseconds);
#else
    ::usleep(FnMilliseconds);
#endif
}
