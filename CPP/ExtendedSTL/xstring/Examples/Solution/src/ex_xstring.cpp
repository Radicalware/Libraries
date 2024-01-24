
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Full.h"
#include "Option.h"
#include "Macros.h"

#include <vld.h>
/// note: google's RE2 has very bad memory leaks, I will need to probably
/// need to switch to pearl regex later and see if that fixes it.
/// to-bad becaues re2 is faster than perl

int CustomTest()
{
    Begin();
    //RA::FormatNum(20686.34404853, 2).Print();
    //RA::TruncateNum(20686.34404853, 4, true).Print();
    //RA::TruncateNum(20686.34404853, 5, true).Print();
    //RA::TruncateNum(20686.34404853, 6, true).Print();
    //RA::TruncateNum(20686.34404853, 7, true).Print();
    //RA::TruncateNum(20686.34404853, 8, true).Print();

    RA::FormatNum(20686.34404853, 3).Print();
    RA::FormatNum(20686.4, 3).Print();
    RA::FormatNum(20686, 3).Print();

    return 0;
    Rescue();
}

void MoveTest()
{
    auto xstr1 = xstring("xtest1");
    auto xstr2 = xstring("xtest2");
    auto sstr = std::string("stest");
    cout << xstr1 << endl;
    cout << xstr2 << endl;
    cout << sstr << endl;
    xstr1 = std::move(sstr);
    cout << sstr << endl;
    cout << xstr1 << endl;
    xstr1 = std::move(xstr2);
    cout << xstr2 << endl;
    cout << xstr1 << endl;
}

int main()
{
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
