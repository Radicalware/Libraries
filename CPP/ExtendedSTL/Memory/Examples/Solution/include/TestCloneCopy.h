#pragma once

#include "Macros.h"
#include "Timer.h"
#include "Mutex.h"
#include "Memory.h"


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
        AccountName = new char[Size + 1];
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
    auto Balances = RA::MakeShared<Balance*>(3, 77, " Riddick "); // 8 indexes & 9 chars
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

