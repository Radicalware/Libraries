
#include <iostream>
#include <string_view>

#include "xstring.h"
#include "xvector.h"
#include "re2/re2.h"

using std::cout;
using std::endl;

void test1()
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
}

void test2()
{
    cout << "===================================\n";
    std::string str;
    std::string data = "XXoneZZ XXtwoZZ";
    re2::RE2::Options ops;
    re2::RE2 rex(R"((XX|ZZ))", ops);
    re2::RE2 rex2(R"((XX|ZZ|YY))", ops);
    re2::StringPiece pc = data;
    int value = 0;


    auto test = [&](int i) {
        size_t loc = RE2::FindAndConsume(&pc, rex, &str);
        cout << "data: " << data << endl;
        cout << "pc:   " << pc << endl;
        cout << "str:  " << str << endl;
        cout << "-------------------\n";
    };
    for (int i = 0; i < 5; i++)
        test(i);

    cout << "orig:  " << data << endl;
    cout << "peice: " << pc << endl;
    cout << "data:  " << data << endl;

}


void test3()
{
    std::string data = "XXoneZZ XXtwoZZ";
    xstring xdata = data;
    xdata.Findall(RE2(R"((?:XX)(\w+)(?:ZZ))")).Join('-').Print(2);
}

int main()
{
    //test1();
    test2();
    //test3();

    // notes: 
    // re2 = slower to create regex object
    // std = faster to create regex object

    // re2 = faster to run scans
    // std = slower to run scans

    // summary, if you scan a lot, use re2, for one-offs, use std

    return 0;
}
