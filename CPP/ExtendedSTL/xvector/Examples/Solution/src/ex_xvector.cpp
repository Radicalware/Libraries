﻿

#include <cctype>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <execution>

int test()
{
    std::string s {"hello"};
    std::transform(s.cbegin(), s.cend(),
        s.begin(), // write to the same location
        [](unsigned char c) { return std::toupper(c); });
    std::cout << "s = " << std::quoted(s) << '\n';

    // achieving the same with std::for_each (see Notes above)
    std::string g {"hello"};
    std::for_each(g.begin(), g.end(), [](char& c) // modify in-place
        {
            c = std::toupper(static_cast<unsigned char>(c));
        });
    std::cout << "g = " << std::quoted(g) << '\n';

    std::vector<char> ordinals;
    ordinals.resize(s.size() + 1);
    std::transform(std::execution::par, s.cbegin(), s.cend(), ordinals.begin(),
        [](unsigned char c) { return std::toupper(static_cast<unsigned char>(c)); });
    ordinals.push_back(0);

    std::cout << "ordinals = " << ordinals.data() << '\n';


    return 0;
}



// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>
#include "xvector.h"
#include "xstring.h"
#include "xmap.h"
#include "Macros.h"
#include "SharedPtr.h"

using std::cout;
using std::endl;

class Abs
{
public:
    typedef Abs value_type; // this is required on Nix
    Abs(){};
    virtual void Print(){}
};

class Der : public Abs
{
public:
    Der(){}
    virtual void Print(){ cout << "Derived" << endl; }
};


void move_vec()
{
    cout << "testing std::vector\n";
    std::vector<xstring> vec1{ "zero","one","two","three","four","five" };
    cout << "vec1.size() = " << vec1.size() << endl;
    xvector<xstring> vec2 = std::move(vec1);
    cout << "vec1.size() = " << vec1.size() << endl;
    cout << "vec2.size() = " << vec2.size() << endl;
}

void move_xvec()
{
    cout << "\ntesting xvector\n";
    xvector<xstring> vec1{ "zero","one","two","three","four","five" };
    cout << "vec1.size() = " << vec1.Length() << endl;
    xvector<xstring> vec2 = std::move(vec1);
    cout << "vec1.size() = " << vec1.Length() << endl;
    cout << "vec2.size() = " << vec2.Length() << endl;

    for (auto& Val : vec1)
        cout << Val << ' ';
    cout << endl;
    for (auto& Val : vec2)
        cout << Val << ' ';
    cout << endl;
    cout << endl;
}

int main(int argc, char** argv) 
{
    Begin();
    Nexus<>::Start();



    auto ListOfInts = xvector<int>({ 1,2,3,4,5 });
    ListOfInts
        .ForEachThreadSeq<xstring>([](const int& El) { return RA::ToXString(El); })
        .Join(',')(0,-1)
        .Print();


    xvector<xp<xstring>> SharedPtrs;
    SharedPtrs.Emplace("hello");
    SharedPtrs.Emplace("again");
    SharedPtrs.Emplace("There");
    SharedPtrs.Join(' ').Print();

    for (auto& Val : SharedPtrs)
        cout << Val.Get() << endl;
    auto SplitPtrs = SharedPtrs.Split(2);
    cout << SplitPtrs[0][1] << endl;

    xstring val = "hello";
    if (SharedPtrs.HasTruth([&val](auto& Item) { return val == Item; })){
        cout << "Has: 'Hello'" << endl;
    }

    if (SharedPtrs.LacksTruth([](auto& Item, xstring Test = "Test") { return Item == Test; })){
        cout << "Does not have: 'Test'" << endl;
    }


    xvector<Abs*> abs = { new Der };
    abs.SetDeleteAllOnExit(true);
    abs[0].Print();

    cout << "-----------------------------------------" << endl;
    move_vec();
    move_xvec();
    cout << "-----------------------------------------" << endl;

    // test r-val ref    
    // std::vector<std::string>
    // xvector<xstring>
    xvector<xstring> vec_str({ "zero","one","two","three","four","five" });
    vec_str.Add("six", "seven", "eight"); // as many as you want.
    vec_str.Join(' ').Print('\n', '\n');

    cout << "========================================\n";
    cout << "testing xvector discovery\n\n";
    xvector<char>    vec_char{ 'v','a','r','c','h','a','r' };

    const char v1[] = "var1";
    const char v2[] = "var2";
    const char v3[] = "var3";
    const char v4[] = "var4";
    xvector<const char*> vec_chara{ v1, v2, v3, v4 };

    xvector<int>   vec_int{ 0,1,2,3,4,5,6 };

    // ----------- AUTO WAY OF CONVERSIONS --------------------------------------------------------------------------
    cout << "xstring : " << vec_str.Join(' ') << endl;
    cout << "chars   : " << vec_char.Join(' ') << endl;
    cout << "char arr: " << vec_chara.Join(' ') << endl;
    cout << "ints    : " << vec_int.Join(' ') << "\n\n";
    // ----------- LONG WAY OF CONVERSIONS --------------------------------------------------------------------------
    cout << "xstring : " << vec_str.Join(' ') << endl;
    cout << "chars   : " << vec_char.Convert<xstring>().Join(' ') << endl;
    cout << "char arr: " << vec_chara.Convert<xstring>([](const char* val) {return xstring(val); }).Join(' ') << endl;
    cout << "ints    : " << vec_int.Convert<xstring>([](int val) {return RA::ToXString(val); }).Join(' ') << endl;

    xstring three = "three";

    xvector<xstring*> vec_str_ptrs = vec_str.GetPtrs();
    cout << "\nprint ptrs: " << vec_str_ptrs.Join(' ') << '\n';
    cout << vec_str_ptrs[0] << endl;
    vec_str_ptrs.ForEachThread([](xstring& Str) { return ">>> " + Str; }).Join('\n').Print();
    cout << "---" << endl;
    vec_str_ptrs.ForEachThreadSeq<xstring>([](const xstring* Str) { return ">>> " + *Str; }).Join('\n').Print();

    xvector<xstring> vec_str_vals = vec_str_ptrs.GetVals();
    cout << "print vals: " << vec_str_vals.Join(' ') << "\n\n";

    RA::PrintAttributes(three);
    cout << "true  => " << vec_str.Has(three) << endl;
    cout << "true  => " << vec_str.HasTruth([&three](const xstring& Item) { return Item == three; }) << endl;
    cout << "      => " << vec_str.Take("(" + three + ")").Join(' ') << endl;
    cout << "true  => " << vec_str.Has("two") << endl;
    cout << "false => " << vec_str.Has("twenty") << endl;

    cout << "========================================\n";
    cout << "testing regex\n\n";
    xvector<xstring> emails{
       "first.last@subdomain.domain.com",
       "realuser@gmail.com",
       "trap@attack.hack.info"
    };

    cout << "false email = " << emails.FullMatchOne(R"([\w\d_]*@[\w\d_]*)") << endl;
    // that failed because it does not match any one email start to finish
    cout << "true  email = " << emails.FullMatchOne(R"(^([\w\d_]*(\.?)){1,2}@([\w\d_]*\.){1,2}[\w\d_]*$)") << endl;;
    // this one matches the first email start to finish so it replies true

    cout << "true email  = " << emails.MatchOne("[\\w\\d_]*@[\\w\\d_]*") << endl;
    // this time the partial email returns true because the regex matches at least a portion
    // of one of the elements in the array.

    xvector<xstring> found_email = emails.Take("^([\\w\\d_]*(\\.?)){1,2}@([\\w\\d_]*\\.){1,2}[\\w\\d_]*$");
    // take the first email that matches that regex

    cout << "\nemails found\n" << found_email.Join('\n') << endl;

    cout << "\nafter email removed\n" << emails.Remove(R"(trap@[\w\d_]*)").Join('\n') << "\n\n";


    xvector<xstring> tripple = { "one","two","three" };
    tripple *= 4;
    tripple.Split(5).Proc([](auto& xvec) {xvec.Join(' ').Print(); return false; /* (false means never break) */});
    cout << '\n';


    auto sub_function_val = [](xstring& elem, xstring& str) -> xstring {
        return elem.Sub("gmail", str);
    };
    auto sub_function_val2 = [](xstring& elem, xstring& junk, xstring& str) -> xstring {
        return elem.Sub("gmail", str);
    };

    // note: lval/rval makes a difference
    auto sub_function_ptr = [](xstring& elem, xstring&& str) -> xstring {
        return elem.Sub("gmail", str);
    };

    xvector<xstring>arr = { "one@gmail.com","two@gmail.com","three@gmail.com" };

    arr.ForEachThread(sub_function_val2, xstring("junk"), xstring("render")).Join("\n").Print(2);

    arr.ForEachThread(
        [](auto& elem, auto& val) 
        {
            return elem.Sub("gmail", val); 
        },
        "new_val"
    ).Join('\n').Print(2);

    xstring x_new = "new_ref";
    arr.ForEachThread([&x_new](auto& elem) {return elem.Sub("gmail", x_new); }).Join('\n').Print(2);

    arr.ForEachThread([](auto& elem, xstring def_replace = "default") {return elem.Sub("gmail", def_replace); }).Join('\n').Print(2);

    xvector<xstring*> arr_ptr = arr.GetPtrs();
    arr_ptr.ForEach([](xstring& elem, xstring str = "mod_ptr1") {return elem.Sub("gmail", str); }).Join('\n').Print();
    arr_ptr.ForEach(sub_function_ptr, xstring("mod_ptr2")).Join('\n').Print();

    arr_ptr.ForEachThread([](xstring& elem, xstring str = "mod_ptr3") {return elem.Sub("gmail", str); }).Join('\n').Print();
    arr_ptr.ForEachThread(sub_function_val, xstring("mod_ptr4")).Join('\n').Print();
    xstring mod5 = "mod_ptr5";
    arr_ptr.ForEachThread([&mod5](xstring& elem) {return elem.Sub("gmail", mod5); }).Join('\n').Print();


    // TODO FIX
    //arr_ptr.ProcThread([&mod5](xstring& elem) {return elem.Sub("gmail", mod5); }); // executes but returns nothing

    cout << "============================================\n";

    xvector<xstring> seven_eight_nine_nine = { "seven","eight","nine","nine" };
    cout << "vec 1: " << vec_str.Join(' ') << endl;
    cout << "vec 2: " << seven_eight_nine_nine.Join(' ') << endl;

    xvector<xstring> vec_common_values = vec_str.GetCommonItems(seven_eight_nine_nine);
    cout << "common values: " << vec_common_values.Join(' ') << endl;


    xvector<xstring> first  = { "one" , "two", "three" };
    xvector<xstring> second = { "four", "five", "six" };

    auto Third = first + second;
    Third.Join(' ').Print();


    RescuePrint();
    Nexus<>::Stop();
    return 0;
}
