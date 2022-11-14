
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Full.h"
#include "Option.h"
#include "Macros.h"

int main()
{
    RA::FormatNum(20686.34404853, 2).Print();
    RA::TruncateNum(20686.34404853, 4, true).Print();
    RA::TruncateNum(20686.34404853, 5, true).Print();
    RA::TruncateNum(20686.34404853, 6, true).Print();
    RA::TruncateNum(20686.34404853, 7, true).Print();
    RA::TruncateNum(20686.34404853, 8, true).Print();


    return 0;
    Begin();
    Nexus<>::Start();
    // NOTE: All test functions are inline to make example reading easier.
    Full full;
    full.Basics();
    full.add_n_join();

    // NOTE: All test functions are inline to make example reading easier.
    Option option;
    option.Split();
    option.Findall();
    option.Search();
    option.Match();
    option.Sub();
    option.char_count();
    option.str_count();

    RescuePrint();
    Nexus<>::Stop();
    return 0;
}
