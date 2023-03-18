
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

int main() 
{
    Begin();
    Nexus<>::Start(); // note: you could just make an instance of type void
    // and it would do the same thing, then when it would go out of scope (the main function)
    // it would automatically get deleted. I did what is above because I like keeping
    // static classes static and instnace classes instance based to not confuse anyone. 
    // -------------------------------------------------------------------------------------

    ObjectMutexHandling();
    BenchmarkNexus();

    //RA::Timer Time;

    //constexpr xint LnLoops = 10000000;
    //RA::Atomic<xint> Val1 = 0;
    //for (xint i = 0; i < LnLoops; i++) {
    //    ++Val1;
    //}
    //cout << Val1 << " " << Time.GetElapsedTimeMilliseconds() << endl;

    //Time.Reset();
    //xint Val2 = 0;
    //std::mutex SimpleMtx;
    //RA::Mutex Mtx;
    //for (xint i = 0; i < LnLoops; i++) {
    //    //auto Lock = std::unique_lock(SimpleMtx);
    //    auto Lock = Mtx.CreateLock([] {return true; });
    //    ++Val2;
    //}
    //cout << Val2 << " " << Time.GetElapsedTimeMilliseconds() << endl;

    RescuePrint();
    Nexus<>::Stop();
    return 0;
}
