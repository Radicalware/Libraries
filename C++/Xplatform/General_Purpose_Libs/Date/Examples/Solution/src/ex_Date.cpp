
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

    xvector<Date> dates = { 
         Date(3, 29, 2020)
        ,Date("3/29/2020")
        ,Date(737515) // total days since AD
    };

    dates[0] += 6;

    cout << '\n';
    for (auto date : dates)
        cout << date << endl;

    for (auto& date : dates)
        date.set_neat(true);

    cout << '\n';
    for (auto date : dates)
        cout << date << endl;

    Nexus<>::Stop();
    return 0;
}
