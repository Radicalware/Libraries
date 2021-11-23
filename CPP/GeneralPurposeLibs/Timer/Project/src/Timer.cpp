
#include "Timer.h"
#include "xstring.h"

#include <chrono>

Timer::Timer() : m_beg(SteadyClock::now()) {   }

void Timer::Reset() {
    m_beg = SteadyClock::now();
}


void Timer::WaitSeconds(pint extent) const
{
    while (This.GetElapsedTimeMilliseconds() < extent)
        Timer::Sleep(1);
}

void Timer::WaitMilliseconds(unsigned long extent) const
{
    while (This.GetElapsedTimeMilliseconds() < extent / static_cast<pint>(1000))
        Timer::Sleep(1);
}

void Timer::Wait(unsigned long extent) const
{
    This.WaitMilliseconds(extent);
}

void Timer::Lap()
{
    m_laps_xv << This.GetElapsedTimeMilliseconds();
}

void Timer::Lap(const xstring& key)
{
    pint val = This.GetElapsedTimeMilliseconds();
    m_laps_xm.AddPair(key, val);
    m_laps_xv.push_back(val);
}

void Timer::Lap(xstring&& key)
{
    Lap(key);
}

void Timer::Clear()
{
    m_laps_xv.clear();
    m_laps_xm.clear();
}

pint Timer::Get(size_t idx) const
{
    return m_laps_xv[idx];
}

pint Timer::Get(const xstring& key) const
{
    return m_laps_xm.at(key);
}

xvector<pint> Timer::GetVector() const
{
    return m_laps_xv;
}

xmap<xstring, pint> Timer::GetMap() const
{
    return m_laps_xm;
}

std::ostream& operator<<(std::ostream& out, const Timer& time)
{
    out << time.GetElapsedTimeMilliseconds();
    return out;
}

void Timer::Sleep(unsigned long FnMilliseconds)
{
#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
    ::Sleep(FnMilliseconds);
#else
    ::usleep(FnMilliseconds);
#endif
}
