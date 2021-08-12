
// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Nexus.h"
#include "Macros.h"

#include <sstream>
//#include "vld.h"


void ThisWillFail()
{
    xstring* PlayerNamePtr = nullptr;
    int Kills = 55;
    int Health = 98.555;
    NullThrow(PlayerNamePtr, SSS("Player is Null with Kills: ", Kills, " and Health: ", 98.555));

}

struct Object
{
    static void TestPointers()
    {
        NewObject(xstring, Name, "Riddick");
        cout << Name << endl;
        DeleteObject(Name);
        // ------------------------------------------
        xstring* NullStrPtr = nullptr;
        GET(NullStr);
        cout << Name << endl;
        delete NamePtr;
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
        ThisWillFail();
    }
    catch (const xstring& Err)
    {
        Err.Print();
    }

    Begin();
    Object::TestPointers();
    Rescue();

    Begin();
    Object Instance;
    Instance.FastCreateAndDestroy();
    Rescue();

    Nexus<>::Stop();
    return 0;
}
