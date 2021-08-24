
// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Nexus.h"
#include "Macros.h"

#include <sstream>
//#include "vld.h"


void BadFuncitonWithSSS()
{
    xstring* PlayerNamePtr = nullptr;
    int Kills = 55;
    int Health = 98.555;
    NullThrow(PlayerNamePtr, SSS("Player is Null with Kills: ", Kills, " and Health: ", 98.555));
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
        Begin()
        NewObject(Name, xstring("Riddick"));
        cout << Name << endl;
        DeleteObject(NamePtr);

        BadFunctionWithGET();
        RescueThrow();
    }

    xstring* TheNamePtr = nullptr;

    void FastCreateAndDestroy()
    {
        RenewObject(TheName, xstring("King"));
        cout << TheName << endl;
        DeleteObject(TheNamePtr);
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

    try
    {
        BadFuncitonWithSSS();
    }
    catch (const xstring& Err)
    {
        Err.Print("\n");
    }

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

    Nexus<>::Stop();
    return 0;
}
