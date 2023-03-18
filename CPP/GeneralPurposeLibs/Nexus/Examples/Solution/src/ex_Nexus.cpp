
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>
#include<exception>

#include<functional>  // for testing only
#include<thread>      // for testing only
#include<type_traits> // for testing only

#include "Nexus.h"
#include "xvector.h"
#include "xstring.h"
#include "xmap.h"
#include "Timer.h"

using std::cout;
using std::endl;
using std::bind;

#include "ObjectMutexHandling.h"
#include "NexusBenchmark.h"
#include "Timer.h"


double BenchSingleThread()
{
    RA::Timer Time;
    for (int i = 1; i < Nexus<>::GetCPUThreadCount() * 3; i++)
        cout << PrimeNumber(10000) << endl;
    cout << __CLASS__ " Time: " << Time.GetElapsedTimeMicroseconds() << endl;
    return Time.GetElapsedTimeMicroseconds();
}

double BenchMultiThread()
{
    RA::Timer Time;
    Nexus<>::Disable();
    for (int i = 1; i < Nexus<>::GetCPUThreadCount() * 3; i++)
        Nexus<>::AddTask(&PrintPrimeNumber, 10000);
    Nexus<>::Enable();
    Nexus<>::WaitAll();
    cout << __CLASS__ " Time: " << Time.GetElapsedTimeMicroseconds() << endl;
    return Time.GetElapsedTimeMicroseconds();
}


int main() 
{
    Begin();
    Nexus<>::Start(); // note: you could just make an instance of type void
    // and it would do the same thing, then when it would go out of scope (the main function)
    // it would automatically get deleted. I did what is above because I like keeping
    // static classes static and instnace classes instance based to not confuse anyone. 
    // -------------------------------------------------------------------------------------

    //ObjectMutexHandling();
    //BenchmarkNexus();


    auto LnSingleTime = BenchSingleThread();
    auto LnMultiTime  = BenchMultiThread();

    cout << "Threading Speed Inc: " << (LnSingleTime / LnMultiTime) << endl;

    RescuePrint();
    Nexus<>::Stop();
    return 0;
}
