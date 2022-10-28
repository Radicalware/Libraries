
#include "Timer.h"
#include "xstring.h"

#include <chrono>

const RA::Timer RA::Timer::StaticClass;

RA::Timer::Timer() : m_beg(SteadyClock::now()) {   }

void RA::Timer::Reset() {
    m_beg = SteadyClock::now();
}


void RA::Timer::WaitSeconds(pint extent)
{
    WaitMilliseconds(extent * 1000);
}

void RA::Timer::WaitMilliseconds(unsigned long extent)
{
    auto ElapsedTime = StaticClass.GetElapsedTimeMilliseconds();
    while (StaticClass.GetElapsedTimeMilliseconds() < (ElapsedTime + extent))
        RA::Timer::Sleep(1);
}

void RA::Timer::Wait(unsigned long extent)
{
    StaticClass.WaitMilliseconds(extent);
}

void RA::Timer::WaitUntil(unsigned long extent, std::function<bool()>&& Function)
{
    while (!Function())
        StaticClass.Sleep(extent);
}

void RA::Timer::PassOrWait(unsigned long TestEveryTimer, unsigned long ExitAnywayTimer, std::function<bool()>&& Function)
{
    const auto ElapsedTime = StaticClass.GetElapsedTimeMilliseconds();
    // break when the function is true or we surpass a point later in time
    while (!Function() && StaticClass.GetElapsedTimeMilliseconds() < (ElapsedTime + ExitAnywayTimer))
        StaticClass.Sleep(TestEveryTimer);
}

void RA::Timer::PassOrWaitSeconds(unsigned long TestEveryTimer, unsigned long ExitAnywayTimer, std::function<bool()>&& Function)
{
    const auto ElapsedTime = StaticClass.GetElapsedTimeSeconds();
    // break when the function is true or we surpass a point later in time
    while (!Function() && StaticClass.GetElapsedTimeSeconds() < (ElapsedTime + ExitAnywayTimer))
        StaticClass.Sleep(TestEveryTimer);
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

void RA::Timer::SleepSeconds(unsigned long FnSeconds)
{
#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
    ::Sleep(FnSeconds * 1000);
#else
    ::usleep(FnSeconds * 1000);
#endif
}
