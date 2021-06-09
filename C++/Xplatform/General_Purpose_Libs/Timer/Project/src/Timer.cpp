
#include "Timer.h"
#include "xstring.h"

#include <chrono>

Timer::Timer() : m_beg(Timer_clock_t::now()) {   }

void Timer::Reset() {
    m_beg = Timer_clock_t::now();
}

double Timer::GetElapsedTime() const {
    return static_cast<double>(std::chrono::duration_cast<Timer_second_t>(Timer_clock_t::now() - m_beg).count());
}

void Timer::WaitSeconds(double extent) const
{
    while (this->GetElapsedTime() < extent)
        Timer::Sleep(1);
}

void Timer::WaitMilliseconds(unsigned long extent) const
{
    while (this->GetElapsedTime() < extent / static_cast<double>(1000))
        Timer::Sleep(1);
}

void Timer::Wait(unsigned long extent) const
{
    this->WaitMilliseconds(extent);
}

void Timer::Lap()
{
    m_laps_xv << this->GetElapsedTime();
}

void Timer::Lap(const xstring& key)
{
    double val = this->GetElapsedTime();
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

double Timer::Get(size_t idx) const
{
    return m_laps_xv[idx];
}

double Timer::Get(const xstring& key) const
{
    return m_laps_xm.at(key);
}

xvector<double> Timer::GetVector() const
{
    return m_laps_xv;
}

xmap<xstring, double> Timer::GetMap() const
{
    return m_laps_xm;
}

std::ostream& operator<<(std::ostream& out, const Timer& time)
{
    out << time.GetElapsedTime();
    return out;
}

void Timer::Sleep(unsigned long extent)
{
#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
    ::Sleep(extent);
#else
    ::usleep(extent);
#endif
}
