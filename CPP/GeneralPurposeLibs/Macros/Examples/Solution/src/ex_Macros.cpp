
// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Nexus.h"
#include "Macros.h"

#include <sstream>
#include <assert.h>
//#include "vld.h"


void BadFuncitonWithSSS()
{
    Begin();
    xstring* PlayerNamePtr = nullptr;
    int Kills = 55;
    int Health = 98.555;
    NullThrow(PlayerNamePtr, SSS("Player is Null with Kills: ", Kills, " and Health: ", 98.555));
    Rescue();
}

void BadFunctionWithGET()
{
    Begin();
    xstring* NamePtr = nullptr;
    GET(Name);
    cout << Name << endl;
    delete NamePtr;
    RescueThrow();
}

struct Object
{
    static void TestPointers()
    {
        Begin();
        NewObj(Name, xstring("Riddick"));
        cout << Name << endl;
        DeleteObj(NamePtr);

        BadFunctionWithGET();
        RescueThrow();
    }

    xstring* TheNamePtr = nullptr;

    void FastCreateAndDestroy()
    {
        Begin();
        RenewObj(TheName, xstring("King"));
        cout << TheName << endl;
        DeleteObj(TheNamePtr);
        Rescue();
    }
};


void StlException()
{
    Begin();
    throw std::exception("We are throwing an STL Execption!");
    RescueThrow();
}

int main() 
{
    Nexus<>::Start(); 

    Begin();
    BadFuncitonWithSSS();
    RescuePrint();

    Begin();
    Object::TestPointers();
    RescuePrint();

    Begin();
    Object Instance;
    Instance.FastCreateAndDestroy();
    RescuePrint();


    Begin();
    StlException();
    RescuePrint();

    return Nexus<>::Stop();
}
