#pragma once

// !!!ADAPTATION CODE!!!
// originator = LearnCPP.com

#include<iostream>
#include<chrono>
// This code was developed on learncpp.com
// I took the code, and made it easy to
// use with shared object.

// On a side note: I give big thanks for all the
// hard work that was put into learncpp.com!!

class Timer
{
private:
	using clock_t = std::chrono::high_resolution_clock;
	using second_t = std::chrono::duration<double, std::ratio<1> >;
	std::chrono::time_point<clock_t> m_beg = clock_t::now();

public:
	Timer();
	void reset();
	double elapsed() const;
};




