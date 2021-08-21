
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "OS.h"
#include "Nexus.h"

using std::cout;
using std::endl;

#include "ex_open_n_delete.h"
#include "ex_file_managment.h"
#include "ex_bash_style.h"

//#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
//    #include<vld.h>
//#endif

// Note: I had a case where the OS dll had issues building
// The fix was to build it as a SHARED DLL and then switch it back to a MODULE DLL

int main() 
{
    Begin();
    Nexus<>::Start();
    // all static OS functions start with an Upper_Case
    ex_open_n_delete();
    ex_file_managment();
    ex_bash_style();

    RescuePrint();
    Nexus<>::Stop();
    return 0;
}
