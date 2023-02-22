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

class Player
{
public:
    Player(const xstring& FsName) : MsPlayerName(FsName) {}
    auto& GetPlayerName() const { return MsPlayerName; }
    virtual void Attack() const = 0;
protected:
    xstring MsPlayerName;
};

class Barbarian : public Player
{
public:
    using Player::Player;
    virtual ~Barbarian() {};
    virtual void Attack() const { cout << "Swing Axe" << endl; }
};

void TestSharedPtr()
{
    MSP(Player, LoPlayer, MKP<Barbarian>("KingKozaK"));
    cout << LoPlayer.GetPlayerName() << endl;
    LoPlayer.Attack();
}

int main()
{
    Nexus<>::Start();
    Begin();

    TestSharedPtr();
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

