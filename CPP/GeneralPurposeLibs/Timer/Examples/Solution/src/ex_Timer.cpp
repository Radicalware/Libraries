
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include <iostream>

#include "Timer.h"

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
#ifdef DLL_EXPORT
#define EXI __declspec(dllexport)
#else
#define EXI __declspec(dllimport)
#endif
#else
#define EXI
#endif

using std::cout;
using std::endl;

int main(int argc, char** argv)
{
    Nexus<>::Start();

    cout << "Wait 1/10 second\n";
    RA::Timer Timer;
    RA::Timer::Sleep(100);
    cout << Timer << "\n\n";

    // -----------------------------------------------------------
    cout << "Reset and wait 2/10 sec, lap and wait 1/10 second\n";
    Timer.Reset();
    
    RA::Timer::Sleep(200);
    Timer.Lap();

    RA::Timer::Sleep(100);

    xstring TmpStr("3/10");
    Timer.Lap(TmpStr);

    //Timer.Lap("3/10");

    cout << Timer.Get(0) << "\n";
    //cout << Timer.Get("3/10") << "\n";
    cout << Timer.Get(1) << "\n\n";
    // -----------------------------------------------------------

    cout << "Pause until we lap at 4/10 sec\n";
    Timer.WaitSeconds(0.4);
    cout << Timer << endl;

    cout << "Pause until we lap at 6/10 sec\n";
    Timer.Wait(600);
    cout << Timer << endl;

    return Nexus<>::Stop();
}
