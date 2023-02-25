#include <iostream>

// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "xvector.h"
#include "xstring.h"
#include "xmap.h"


#include <map>
#include <unordered_map>

#include <utility>

using std::cout;
using std::endl;
using std::string;

// note: enum   = text to num
//       vector = num  to text
// only vector can go both ways
// we reference vector value by int
// if we need the other way around we use a map (string to int)
// because of that, xrender default return type is string due to .join

int main()
{   
    Nexus<>::Start();

    xmap<xstring, xp<xstring>> MapPtrs;

    std::map<xstring, int> smap1 = { { "one", 1}, { "two", 2 }, { "Three", 3 } };
    std::unordered_map<xstring, int> sumap1 = { { "one", 1}, { "two", 2 }, { "Three", 3 } };
    xmap<xstring, int> xxmap = { { "one", 1}, { "two", 2 }, { "Three", 3 } };

    cout << "xxmap.size()  = " << xxmap.size() << '\n';
    xmap<xstring, int> xmap1 = std::move(xxmap);
    cout << "xxmap.size()  = " << xxmap.size() << '\n';
    cout << "xmap1.size()  = " << xmap1.size() << "\n\n";

    cout << "smap1.size()  = " << smap1.size() << '\n';
    xmap<xstring, int> xmap2 = std::move(smap1);
    cout << "smap1.size()  = " << smap1.size() << '\n';
    cout << "xmap2.size()  = " << xmap2.size() << "\n\n";

    cout << "sumap1.size() = " << sumap1.size() << '\n';
    xmap<xstring, int> xmap3  = std::move(sumap1);
    cout << "sumap1.size() = " << sumap1.size() << '\n';
    cout << "xmap3.size()  = " << xmap3.size() << "\n\n";

    xvector<xstring> single_vec{ "one","two","three","four","five","six" };
    xvector<xvector<xstring>> double_vec = single_vec.Split(3);

    single_vec.SubAll("(o)", "0").Join(' ').Print();

    single_vec.Remove("(^f)").Join(' ').Print();
    single_vec.Take("(^f)").Join(' ').Print();

    cout << "\n========================================\n";
    cout << "testing map discovery\n\n";

    xmap<xstring, xvector<xstring>> map_s_v = { 
        {"V-First", {"1-one","1-two","1-three"}},
        {"V-Second",{"2-one","2-two","2-three"}}
    };

    xmap<xstring, xstring> smap = {
        { "hash1111", "AAAA"},
        { "hash222", "BBBB"},
        { "hash33", "CCCC"},
        { "hash4", ""}
    };

    map_s_v.GetKeys().Join(' ').Print();
    smap.GetKeys().Join(' ').Print();

    // the following three are the same method but with different verbage
    cout << "1.) map_ss_1 value for 'hash222' = " << smap["hash222"] << endl;
    cout << "2.) map_ss_1 value for 'hash222' = " << smap.Key("hash222") << endl;
    cout << "3.) map_ss_1 value for 'hash222' = " << smap.GetValueFrom("hash222") << endl;

    cout << "its value is BBBB = " << smap.GetValueFrom("hash222").Is("BBBB") << endl; // true
    cout << "its value is BBBB = " << smap.Key("hash222").Is("BBBB") << endl; // true
    cout << "its value is BBBB = " << smap("hash222", "BBBB") << endl; // true, alternate method
    try {
        cout << "its value is BBBB = " << smap("I don't exist", "BBBB") << endl; // false
    }
    catch (...)
    {
        cout << "Exception Caught\n";
    }
    
    xstring key_values1;
    smap.Proc([&key_values1](const auto& key, const auto& value) {
        if (value.size())
            key_values1 += value + " ";
        return false; // never break loop
    });
    cout << '[' << key_values1 << ']' << endl;

    // note smap.ForEachThread<string> is not needed because we are returning xvector<key type>
    xstring key_values2 = smap.ForEachThread([](const auto& key, const auto& value) {
        xstring ret_str;
        if (value.size())
            ret_str += value + " ";
        return ret_str;
    }).Join();
    cout << '[' << key_values2 << ']' << endl;

    cout << "========================================\n";

    xvector<xstring> str_lst = { "one","two","three","four","five" };
    xvector<xstring*> ptr_lst = str_lst.GetPtrs();

    xmap<xstring*, xstring> map_ptrs;
    int count = 1;
    for (xstring* ptr : ptr_lst) {
        map_ptrs.AddPair(ptr, xstring(count, '='));
        count++;
    }

    map_ptrs.AllocateKeys()->Join('\n').Print(2);

    cout << "cached truth: " << map_ptrs.GetCache().Has(str_lst[0]) << endl; // faster when cached
    cout << "slow   truth: " << map_ptrs.Has(str_lst[0]) << endl;

    cout << "========================================\n";

    smap.Print(2);

    xmap<xstring, int> awd_cars = {
        { "Ford RS", 350 },
        { "VW Golf R", 288 },
        { "Subaru WRX STI", 310 },
        { "Audi S5", 349 }
    };

    awd_cars.ForEachThread([](const xstring& key, const int& value) {
        return key + " = " + RA::ToXString(value) + " HP";
    }).Join('\n').Print(2);

    xmap<xstring, xstring> tmp_mp;
    std::map<xstring, xstring> std_map = smap.ToStdMap(); 
    tmp_mp = std_map;
    std::unordered_map<xstring, xstring> std_u_map = smap.ToStdUnorderedMap(); 
    tmp_mp = std_u_map;
    tmp_mp.Print();


    Nexus<>::Stop();
    return 0;
}

