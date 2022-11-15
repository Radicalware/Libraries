﻿#include <iostream>

// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Macros.h"
#include "Timer.h"
#include "Mutex.h"

#include "StrAtomic.h"

#include "Bench/NoAtomic.h"
#include "Bench/FundamentalAtomic.h"
#include "Bench/ClassAtomic.h"


xstring BenchAtomicClass()
{
    Begin();
    auto RunBench = []()
    {
        Bench::ClassAtomic Obj;
        RA::Timer t;
        RA::SharedPtr<RA::Mutex> LoMutex; // REMOVE REMOVE REMOVE REMOVE REMOVE
        for (int i = 0; i < Obj.MnLooper; i++) {
            if (i % 3 == 0) { // occurs 1/3 of the time
                Nexus<>::AddJob(LoMutex, Obj, &Bench::ClassAtomic::AddValue, -2);
            }
            else {            // occurs 2/3 of the time
                Nexus<>::AddJob(LoMutex, Obj, &Bench::ClassAtomic::AddValue, 2);
            }
            //printf("Count: %u\r", i);
        }
        Nexus<>::WaitAll();
        std::stringstream SS;
        SS << "BenchAtomicClass()" << " Completion Time : " << t.GetElapsedTime() << endl;
        SS << "BenchAtomicClass()" << " Resulting Value : " << Obj.Get() << endl;
        SS << "\n";

        cout << "Done: BenchAtomicClass()\n";
        return std::make_tuple(SS.str(), Obj.Get());
    };


    auto [SS1, Val1] = RunBench();
    auto [SS2, Val2] = RunBench();

    xstring Equality;
    if (Val1 != Val2) Equality += "Not Equal\n\n";
    else              Equality += "Equal\n\n";

    xstring RetStr;;
    RetStr += SS1;
    RetStr += SS2;
    RetStr += Equality;
    return RetStr;
    Rescue();
}

xstring BenchAtomicFundamental()
{
    Begin();
    auto RunBench = []()
    {
        Bench::FundamentalAtomic Obj;
        RA::Timer t;
        RA::SharedPtr<RA::Mutex> LoMutex; // REMOVE REMOVE REMOVE REMOVE REMOVE
        for (int i = 0; i < Obj.MnLooper; i++) {
            if (i % 3 == 0) { // occurs 1/3 of the time
                Nexus<>::AddJob(LoMutex, Obj, &Bench::FundamentalAtomic::AddValue, -2);
            }
            else {            // occurs 2/3 of the time
                Nexus<>::AddJob(LoMutex, Obj, &Bench::FundamentalAtomic::AddValue, 2);
            }
            //printf("Count: %u\r", i);
        }
        Nexus<>::WaitAll();
        std::stringstream SS;
        SS << "BenchAtomicFundamental()" << " Completion Time : " << t.GetElapsedTime() << endl;
        SS << "BenchAtomicFundamental()" << " Resulting Value : " << Obj.Get() << endl;
        SS << "\n";

        cout << "Done: BenchAtomicFundamental()\n";
        return std::make_tuple(SS.str(), Obj.Get());
    };


    auto [SS1, Val1] = RunBench();
    auto [SS2, Val2] = RunBench();

    xstring Equality;
    if (Val1 != Val2) Equality += "Not Equal\n\n";
    else              Equality += "Equal\n\n";

    xstring RetStr;;
    RetStr += SS1;
    RetStr += SS2;
    RetStr += Equality;
    return RetStr;
    Rescue();
}

xstring BenchSharedPtr()
{
    Begin();
    auto RunBench = []()
    {
        Bench::NoAtomic Obj;
        RA::Timer t;
        RA::SharedPtr<RA::Mutex> LoMutex; // REMOVE REMOVE REMOVE REMOVE REMOVE
        for (int i = 0; i < Obj.MnLooper; i++) {
            if (i % 3 == 0) { // occurs 1/3 of the time
                Nexus<>::AddJob(LoMutex, Obj, &Bench::NoAtomic::AddValue, -2);
            }
            else {            // occurs 2/3 of the time
                Nexus<>::AddJob(LoMutex, Obj, &Bench::NoAtomic::AddValue, 2);
            }
            //printf("Count: %u\r", i);
        }
        Nexus<>::WaitAll();
        std::stringstream SS;
        SS << "BenchSharedPtr()" << " Completion Time : " << t.GetElapsedTime() << endl;
        SS << "BenchSharedPtr()" << " Resulting Value : " << Obj.Get() << endl;
        SS << "\n";

        cout << "Done: BenchSharedPtr()\n";
        return std::make_tuple(SS.str(), Obj.Get());
    };

    auto [SS1, Val1] = RunBench();
    auto [SS2, Val2] = RunBench();

    xstring Equality;
    if (Val1 != Val2) Equality += "Not Equal\n\n";
    else              Equality += "Equal\n\n";

    xstring RetStr;;
    RetStr += SS1;
    RetStr += SS2;
    RetStr += Equality;
    return RetStr;
    Rescue();
}

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


    if(true) // true, false
    {
        Nexus<xstring> Nex;
        Nex.AddJob("BenchAtomicClass",       BenchAtomicClass);
        Nex.AddJob("BenchAtomicFundamental", BenchAtomicFundamental);
        Nex.AddJob("BenchSharedPtr",         BenchSharedPtr);

        cout << "\n\n";
        for (xstring& Output : Nex.GetAll())
            cout << Output << endl;
    }

    RescuePrint();
    Nexus<>::Stop();
    return 0;
}
