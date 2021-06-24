
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

    {
        std::time_t result = std::time(nullptr);
        std::cout << std::asctime(std::localtime(&result)) << result << " seconds since the Epoch\n";
    }
    {

        std::time_t result(1624345800);
        std::cout << std::asctime(std::localtime(&result)) << result << " seconds since the Epoch\n";
    }

    Date LoDate(1624345800);

    cout << LoDate << endl;

    //xvector<Date> dates = { 
    //     Date(3, 29, 2020)
    //    ,Date("3/29/2020")
    //    ,Date(737515) // total days since AD
    //};

    //dates[0] += 6;

    //cout << '\n';
    //for (auto date : dates)
    //    cout << date << endl;

    //for (auto& date : dates)
    //    date.SetNeat(true);

    //cout << '\n';
    //for (auto date : dates)
    //    cout << date << endl;

    Nexus<>::Stop();
    return 0;
}
