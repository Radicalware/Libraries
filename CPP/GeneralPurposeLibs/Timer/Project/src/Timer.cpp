
#include "Timer.h"
#include "xstring.h"

#include <chrono>

const RA::Timer RA::Timer::StaticClass;

RA::Timer::Timer() : MoClock(SteadyClock::now()) {   }

void RA::Timer::Reset() {
    MoClock = SteadyClock::now();
}

xint RA::Timer::GetElapsedTimeSeconds() const {
    return static_cast<xint>(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - MoClock).count());
}

xint RA::Timer::GetElapsedTimeMilliseconds() const {
    return static_cast<xint>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - MoClock).count());
}

xint RA::Timer::GetElapsedTime() const {
    return static_cast<xint>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - MoClock).count());
}

xint RA::Timer::GetElapsedTimeMicroseconds() const {
    return static_cast<xint>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - MoClock).count());
}

void RA::Timer::WaitSeconds(xint extent)
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

#ifndef UsingNVCC
void RA::Timer::Lap()
{
    MvLaps << GetElapsedTimeMilliseconds();
}

void RA::Timer::Lap(const xstring& key)
{
    xint LnTime = GetElapsedTimeMilliseconds();
    MmLaps.AddPair(key, LnTime);
    MvLaps.Add(LnTime);
}

void RA::Timer::Lap(xstring&& key)
{
    Lap(key);
}

void RA::Timer::Clear()
{
    MvLaps.clear();
    MmLaps.clear();
}

xint RA::Timer::Get(size_t idx) const
{
    return MvLaps[idx];
}

xint RA::Timer::Get(const xstring& key) const
{
    return MmLaps.Key(key);
}

xvector<xint> RA::Timer::GetVector() const
{
    return MvLaps;
}

xmap<xstring, xint> RA::Timer::GetMap() const
{
    return MmLaps;
}
#endif // !UsingNVCC

std::ostream& operator<<(std::ostream& out, const RA::Timer& time)
{
    out << time.GetElapsedTimeMilliseconds();
    return out;
}

