#include <iostream>

// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Macros.h"
#include "Timer.h"
#include "Mutex.h"
#include "Memory.h"


#include "StrAtomic.h"
#include "Bench/NoAtomic.h"
#include "Bench/FundamentalAtomic.h"
#include "Bench/ClassAtomic.h"

#include <memory>
#include <vld.h>


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

struct Balance
{
    Balance() {
        cout << "Constructor Called" << endl;
        //AccountName = new char[10];
        //memcpy(AccountName, "KingKozaK\0", 10);
    }
    ~Balance() {
        cout << "Destructor Called" << endl;
        DeleteArr(AccountName);
    }

    void operator=(const Balance& Other)
    {
        CryptoID = Other.CryptoID;
        Amount = Other.Amount;

        Construct(Other.Multiplier, Other.AccountName);
    }

    void Construct(int FnMultiplier, const char* FsAccountname)
    {
        Multiplier = FnMultiplier;
        DeleteArr(AccountName);
        auto Size = strlen(FsAccountname);
        AccountName = new char[Size+1];
        strncpy(AccountName, FsAccountname, Size);
        AccountName[Size] = 0;
    }

    uint CryptoID = 335;
    double Amount = 3453.57839;
    int Multiplier = 0;

    char* AccountName = nullptr;
};

void TestClone()
{
#if _HAS_CXX20
    auto Balances = RA::MakeShared<Balance[]>(3, 77, " Riddick ");// .Proc([](Balance& Elem) { Elem.Multiplier = 77; });
    RA::SharedPtr<Balance[]> CloneBalance = nullptr;
#else
    auto Balances = RA::MakeShared<Balance*>(3, 77,  " Riddick "); // 8 indexes & 9 chars
    RA::SharedPtr<Balance*> CloneBalance = nullptr;
#endif
    CloneBalance.Clone(Balances);

    for (auto& Balance : CloneBalance) // proves we have a true copy
    {
        Balance.AccountName[0] = '-';
        Balance.AccountName[8] = '-'; 
        // 8  = correct
        // 9  = overwrite null end
        // 10 = segfault
    }

    for (auto& Balance : Balances)
    {
        cout << "Holder: " << Balance.AccountName << endl;
        cout << Balance.CryptoID << endl;
        cout << Balance.Amount << endl;
        cout << Balance.Multiplier << endl << endl;
    }
    for (auto& Balance : CloneBalance)
    {
        cout << "Holder: " << Balance.AccountName << endl;
        cout << Balance.CryptoID << endl;
        cout << Balance.Amount << endl;
        cout << Balance.Multiplier << endl << endl;
    }
    cout << "TestClone Done!" << endl;
}


#if _HAS_CXX20
bool BxSame(RA::SharedPtr<int[]> SPtr, xvector<int> VecInts)
#else
bool BxSame(RA::SharedPtr<int*> SPtr, xvector<int> VecInts)
#endif
{
    for (int i = 0; i < 10; i++)
    {
        if (SPtr[i] != VecInts[i])
            return false;
    }
    return true;
}

void TestCopy()
{
    Begin();
#if _HAS_CXX20
    auto SPtr = xp<int[]>(10);
#else
    auto SPtr = xp<int*>(10);
#endif
    xvector<int> VecInts;
    for (int i = 0; i < 10; i++)
        VecInts << i;

    for (int i = 0; i < 10; i++)
        SPtr[i] = VecInts[i];

    for (auto El : SPtr)
        cout << El << ' ';
    cout << endl;
    const auto LbSame = BxSame(SPtr, VecInts);
    cout << endl;
    cout << "Same Copy: " << LbSame << endl;
    Rescue();
}


struct Num
{
    int Val = 0;
};

void ReferencePtrTest()
{
    Begin();

    auto First  = MKP<Num>();
    auto Second = MKP<Num>();

    auto Used = rp<Num>();
    auto UsedReF = Used;

    First->Val  = 1;
    Second->Val = 2;

    Used = First;
    cout << Used->Val << endl;
    cout << UsedReF->Val << endl;

    Used = Second;
    cout << Used->Val << endl;
    cout << UsedReF->Val << endl;

    Rescue();
}

int main()
{
    Nexus<>::Start();
    Begin();
    TestClone();
    TestCopy();
    ReferencePtrTest();

    if(false) // true, false
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
    return Nexus<>::Stop();
}

