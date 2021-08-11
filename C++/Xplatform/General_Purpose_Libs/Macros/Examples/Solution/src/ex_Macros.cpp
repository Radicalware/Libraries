
// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Nexus.h"
#include "Macros.h"


void ThisWillFail()
{
    xstring* NullStr = nullptr;

    double dbl = 3.3333;
    xstring str = "str-val";
    NullThrow(NullStr);
    //NullThrow(NullStr, SSS("this Number three: ", 3, " and double ", dbl, " and str = ", str));

}

void TestPointers()
{
    xstring* NamePtr = new xstring("Riddick");
    GET(Name);
    cout << Name << endl;
    delete NamePtr;
    // ------------------------------------------
    xstring* NullStrPtr = nullptr;
    GET(NullStr);
    cout << Name << endl;
    delete NamePtr;
}

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
    TestPointers();
    Rescue();

    Nexus<>::Stop();
    return 0;
}
