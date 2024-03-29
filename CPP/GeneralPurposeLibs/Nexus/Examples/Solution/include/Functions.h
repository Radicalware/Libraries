
#include "Macros.h"



using ull = unsigned long long;

template<typename F>
void FunctionalExamples(Nexus<ull>& FoNexUll, Nexus<xstring>& FoNexStr, F&& FfPrimeNumber)
{
    Begin();
    cout << "Jobs in progress should not exceed: " << FoNexUll.GetCPUThreadCount() << endl;
    cout << "Jobs In Progress: " << FoNexUll.GetUsedThreadCount() << endl;
    cout << "Starting Loop\n";

    cout << "waiting for jobs to finish\n";

    FoNexUll.WaitAll();
    FoNexStr.Clear();
    FoNexStr.SetAllowedThreadCount(FoNexStr.GetCPUThreadCount() - 1);
    cout << "CPU Thread Count: " << FoNexUll.GetCPUThreadCount() << endl;

    for (int i = 1; i < FoNexUll.GetCPUThreadCount() * 2; i++) 
    {
        FoNexUll.AddTask(std::to_string(i), FfPrimeNumber, i * 888);
        cout << "Jobs Running: " << FoNexUll.GetUsedThreadCount() << endl;
    }
    FoNexUll.ResetAllowedThreadCount();
    cout << "CPU Thread Count: " << FoNexUll.GetCPUThreadCount() << endl;

    FoNexUll.WaitAll(); // wait all isn't required because the getter will cause the wait
    // but the print will be smoother if you wait for all the values to be populated first

    for (int i = 1; i <= FoNexUll.Size(); i++) {
        cout << FoNexUll(i) << endl;
    }

    Nexus<void>::AddTask(
        []()->void {
        cout << "Examples Done!!" << endl;
        });
    Nexus<>::WaitAll();
    Rescue();
}