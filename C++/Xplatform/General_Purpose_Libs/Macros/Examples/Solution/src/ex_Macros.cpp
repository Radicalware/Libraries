
// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Nexus.h"
#include "Macros.h"


void ThisWillFail()
{
    xstring* NullStr = nullptr;

    double dbl = 3.3333;
    xstring str = "str-val";
    NullThrow(NullStr);

}

struct Object
{
    static void TestPointers()
    {
        CreateObject(xstring, Name, "Riddick");
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
        CreateClassObject(xstring, TheName, "King");
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
