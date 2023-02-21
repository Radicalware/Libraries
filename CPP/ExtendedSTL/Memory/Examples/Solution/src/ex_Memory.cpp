#include <iostream>

// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Macros.h"
#include "Timer.h"
#include "Mutex.h"
#include "Memory.h"

#include "Benchmark.h"
#include "ReferenePtrTest.h"
#include "TestCloneCopy.h"
#include "AbstractTest.h"

#include <memory>
#include <vld.h>

void TestFunction(xstring&& Input)
{
    Input.Print();
    Input += " appended";
    Input.Print();
}

int main()
{
    Nexus<>::Start();
    Begin();

    TestClone();
    TestCopy();
    ReferencePtrTest::Run();
    Abstract::Run();

    if(false) // true, false
    {
        Nexus<xstring> Nex;
        Nex.AddJob("BenchAtomicClass",       Benchmark::AtomicClass);
        Nex.AddJob("BenchAtomicFundamental", Benchmark::AtomicFundamental);
        Nex.AddJob("BenchSharedPtr",         Benchmark::SharedPtr);

        cout << "\n\n";
        for (xstring& Output : Nex.GetAll())
            cout << Output << endl;
    }

    RescuePrint();
    return Nexus<>::Stop();
}

