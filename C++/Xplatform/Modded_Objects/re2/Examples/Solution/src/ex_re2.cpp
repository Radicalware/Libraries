
#include <iostream>
#include <string_view>

#include "xstring.h"
#include "xvector.h"
#include "re2/re2.h"

using std::cout;
using std::endl;

int main()
{

    cout << "===================================\n";
    std::string str;
    std::string data = "XXoneZZ XXtwoZZ";
    re2::StringPiece pc = data;

    RE2::FindAndConsume(&pc, RE2(R"((?:XX)(\w+)(?:ZZ))"), &str);
    cout << str << endl;

    RE2::FindAndConsume(&pc, RE2(R"((?:XX)(\w+)(?:ZZ))"), &str);
    cout << str << endl;

    cout << "orig:  " << data << endl;
    cout << "peice: " << pc << endl;
    cout << "data:  " << data << endl;

    xstring xdata = data;
    xdata.findall(RE2(R"((?:XX)(\w+)(?:ZZ))")).join('\n').print();

    // notes: 
    // re2 = slower to create regex object
    // std = faster to create regex object

    // re2 = faster to run scans
    // std = slower to run scans

    // summary, if you scan a lot, use re2, for one-offs, use std

    return 0;
}
