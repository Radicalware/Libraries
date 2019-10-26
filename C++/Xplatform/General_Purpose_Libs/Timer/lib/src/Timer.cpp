
#include "Timer.h"

#include <chrono>

Timer::Timer() : m_beg(Timer_clock_t::now()) {   }

void Timer::reset() {
	m_beg = Timer_clock_t::now();
}

double Timer::elapsed() const {
	return std::chrono::duration_cast<Timer_second_t>(Timer_clock_t::now() - m_beg).count();
}

void Timer::wait_seconds(double extent) const
{
    while (this->elapsed() < extent)
        this->sleep(1);
}

void Timer::wait_milliseconds(unsigned long extent) const
{
    while (this->elapsed() < extent / static_cast<double>(1000))
        this->sleep(1);
}

void Timer::wait(unsigned long extent) const
{
    this->wait_milliseconds(extent);
}

void Timer::lap()
{
    m_laps_xv << this->elapsed();
}

void Timer::lap(const xstring& key)
{
    double val = this->elapsed();
    m_laps_xm.add_pair(key, val);
    m_laps_xv.push_back(val);
}

void Timer::clear()
{
    m_laps_xv.clear();
    m_laps_xm.clear();
}


void Timer::sleep(unsigned long extent) const
{
#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
    ::Sleep(extent);
#else
    ::usleep(extent);
#endif
}

double Timer::get(size_t idx) const
{
    return m_laps_xv[idx];
}

double Timer::get(const xstring& key) const
{
    return m_laps_xm.at(key);
}

xvector<double> Timer::get_xvector() const
{
    return m_laps_xv;
}

xmap<xstring, double> Timer::get_xmap() const
{
    return m_laps_xm;
}

std::ostream& operator<<(std::ostream& out, const Timer& time)
{
    out << time.elapsed();
    return out;
}
