#pragma once

#include "IntervalMap.h"

namespace Example
{
    void Interval()
    {
        auto LoInterval = IntervalMap<int, char>('A');
        LoInterval.Assign(01, 02, 'B');
        LoInterval.Assign(05, 10, 'C');
        LoInterval.Assign(20, 30, 'D');
        LoInterval.Close();
        LoInterval.PrintMap();
        cout << "----------------------" << endl;
        cout << 'C' << " : " << LoInterval['C'] << endl;

        auto LvTests = std::vector{ 1,2,7,25,100 };
        for(const auto& LnVal : LvTests)
            cout << LnVal << " >> " << LoInterval.GetValue(LnVal).first << ":" << LoInterval.GetValue(LnVal).second << endl;
    }
}
