
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
        NewObject(xstring, Name, "Riddick");
        cout << Name << endl;
        DeleteObject(Name);


        BadFunctionWithGET();
        RescueThrow();
    }

    xstring* TheNamePtr = nullptr;

    void FastCreateAndDestroy()
    {
        RenewObject(xstring, TheName, "King");
        cout << TheName << endl;
        DeleteObject(TheName);
    }
};


int main() 
{
    Nexus<>::Start(); 

    try
    {
        BadFuncitonWithSSS();
    }
    catch (const xstring& Err)
    {
        Err.Print();
    }

    Begin();
    Object::TestPointers();
    RescuePrint();

    Begin();
    Object Instance;
    Instance.FastCreateAndDestroy();
    RescuePrint();

    Nexus<>::Stop();
    return 0;
}
