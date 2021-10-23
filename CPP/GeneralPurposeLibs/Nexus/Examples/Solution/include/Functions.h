
#include "Macros.h"



using ull = unsigned long long;

template<typename F>
void FunctionalExamples(Nexus<ull>& FoNexUll, Nexus<xstring>& FoNexStr, F&& FfPrimeNumber)
{
    cout << "Jobs in progress should not exceed: " << FoNexUll.GetCPUThreadCount() << endl;
    cout << "Jobs In Progress: " << FoNexUll.GetThreadCountUsed() << endl;
    cout << "Starting Loop\n";

    cout << "waiting for jobs to finish\n";

    FoNexUll.WaitAll();
    FoNexStr -= 1; // decrease allowed threads by 1
    cout << "Usable Threads: " << FoNexUll.GetCPUThreadCount() << endl;

    for (int i = 0; i < FoNexUll.GetCPUThreadCount() * 2; i++) {
        FoNexUll.AddJob(FfPrimeNumber, 10000);
        FoNexUll.Sleep(5); // a thread isn't listed as being "used" until the actual process starts
        // not when the "add_job" function is executed because that process may be just sitting in a queue
        cout << "Jobs Running: " << FoNexUll.GetThreadCountUsed() << endl;
    }
    FoNexUll.ResetAllowedThreadCount();
    cout << "Usable Threads: " << FoNexUll.GetCPUThreadCount() << endl;

    FoNexUll.WaitAll(); // wait all isn't required because the getter will cause the wait
    // but the print will be smoother if you wait for all the values to be populated first
    for (int i = 0; i < FoNexUll.Size(); i++) {
        cout << FoNexUll(i) << endl;
    }

    Nexus<>::AddJob([]()->void {
        cout << "Examples Done!!" << endl;
        });
    Nexus<>::WaitAll();
}