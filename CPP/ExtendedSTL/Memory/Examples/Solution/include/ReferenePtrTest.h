#pragma once

#include "Macros.h"
#include "Timer.h"
#include "Mutex.h"
#include "Memory.h"


namespace ReferencePtrTest
{
    struct Num
    {
        int Val = 0;
    };

    void Run()
    {
        Begin();

        auto First = MKP<Num>();
        auto Second = MKP<Num>();

        auto Used = rp<Num>();
        auto UsedReF = Used;

        First->Val = 1;
        Second->Val = 2;

        Used = First;
        cout << Used->Val << endl;
        cout << UsedReF->Val << endl;

        Used = Second;
        cout << Used->Val << endl;
        cout << UsedReF->Val << endl;

        Rescue();
    }

}
