
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
    Timer timer;
    Timer::Sleep(100);
    cout << timer << "\n\n";

    // -----------------------------------------------------------
    cout << "Reset and wait 2/10 sec, lap and wait 1/10 second\n";
    timer.Reset();
    
    Timer::Sleep(200);
    timer.Lap();

    Timer::Sleep(100);

    xstring TmpStr("3/10");
    timer.Lap(TmpStr);

    //timer.Lap("3/10");

    cout << timer.Get(0) << "\n";
    //cout << timer.Get("3/10") << "\n";
    cout << timer.Get(1) << "\n\n";
    // -----------------------------------------------------------

    cout << "Pause until we lap at 4/10 sec\n";
    timer.WaitSeconds(0.4);
    cout << timer << endl;

    cout << "Pause until we lap at 6/10 sec\n";
    timer.Wait(600);
    cout << timer << endl;

    return Nexus<>::Stop();
}
