
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>


#include "Nexus.h"
#include "xvector.h"
#include "xstring.h"
#include "Date.h"

using std::cout;
using std::endl;

int main() 
{
    Nexus<>::Start();

    //Date LoDate(1624345800);

    //cout << "Date: " << LoDate << endl;

    //{
    //    std::time_t result = std::time(nullptr);
    //    std::cout << std::asctime(std::localtime(&result)) << result << " seconds since the Epoch\n";
    //}
    //{

    //    std::time_t result(1624345800);
    //    std::cout << std::asctime(std::localtime(&result)) << result << " seconds since the Epoch\n";
    //}


    Date LoDate1(1624345800);

    cout << LoDate1 << endl;

    cout << LoDate1.GetStr() << endl;

    Date LoDate2(LoDate1.GetStr());

    cout << LoDate2 << endl;

    Nexus<>::Stop();
    return 0;
}
