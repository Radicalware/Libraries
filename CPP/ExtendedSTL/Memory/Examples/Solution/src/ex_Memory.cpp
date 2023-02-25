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
    INL     auto&   GetPlayerName() const { return MsPlayerName; }
    virtual xstring Attack() const = 0;
    TTT INL T&      Cast() { return *dynamic_cast<T*>(this); }
protected:
    xstring MsPlayerName;
};

class Barbarian : public Player
{
public:
    using Player::Player;
    INL virtual         ~Barbarian() {};
    INL virtual xstring Attack() const { return "Swing Axe"; }
    INL         auto&   GetArmor() const { return MsArmor; }
    INL         void    SetArmor(const xstring& Str) { MsArmor = Str; }
protected:
    xstring MsArmor;
};

xp<Player> GetBarbarian(const xstring& FsName)
{
    MSP(Player, LoPlayer, MKP<Barbarian>(FsName));
    return LoPlayerPtr;
}

void TestSharedPtr()
{
    Begin();
    Nexus<xp<Player>> Nex;
    Nex.AddJob(&GetBarbarian, "KingKozaK");
    Nex.WaitAll();

    xvector<xp<Player>> Players = Nex.GetMoveAllIndices();
    auto& LoPlayer = Players.First();


    cout << LoPlayer.GetPlayerName() << endl;
    cout << LoPlayer.Attack() << endl;
            LoPlayer.Cast<Barbarian>().SetArmor("Light");
    cout << LoPlayer.Cast<Barbarian>().GetArmor() << endl;
    Rescue();
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

