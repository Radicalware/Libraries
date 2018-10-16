#include<iostream>
#include<chrono>
using namespace std;

#include "./Timer.h"
// Lib by learncpp.com

Timer::Timer() : m_beg(clock_t::now())
{   }

void Timer::reset(){ 
    m_beg = clock_t::now(); 
}

double Timer::elapsed() const  { 
    return std::chrono::duration_cast<second_t>(clock_t::now() - m_beg).count();    
}

