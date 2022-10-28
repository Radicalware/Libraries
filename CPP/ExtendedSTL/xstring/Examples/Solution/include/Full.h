#pragma once

// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>

#include "xstring.h"
#include "xvector.h"
#include<iostream>

using std::string;
using std::cout;
using std::endl;


struct Full
{
    // NOTE: All test functions are inline to make example reading easier.

    template<typename T>
    inline void Print(T val) {
        cout << val << endl;
    }

    inline void Reset(const xvector<xstring>& saved1, const xvector<xstring>& saved2,
        xvector<xstring>& throwback1, xvector<xstring>& throwback2)
    {
        throwback1 = saved1;
        throwback2 = saved2;
    }

    void move_std_string()
    {
        std::string std_string = "test std::string";
        cout << "std_string.size() = " << std_string.size() << '\n';
        xstring x_string = std::move(std_string);
        cout << "std_string.size() = " << std_string.size() << '\n';
        cout << "x_string.size()   = " << x_string.size() << "\n\n";

    }

    void move_xstring()
    {
        xstring x1_string = "test xstring";
        cout << "x1_string.size()   = " << x1_string.size() << '\n';
        xstring x2_string = std::move(x1_string);
        cout << "x1_string.size()   = " << x1_string.size() << '\n';
        cout << "x2_string.size()   = " << x2_string.size() << "\n\n";
    }

    inline int Basics() 
    {
        move_std_string();
        move_xstring();

        xstring hello_world("Hello World");
        Print(hello_world.Split(' ').Join(" ** "));

        Print(xstring("Result 'World' Found: ") + RA::ToXString(hello_world.Scan("[wW][0oO]rld$")));
        Print(xstring("Result 'World' Found: ") + RA::ToXString(hello_world.Match("^.*[wW][0oO]rld$")));

        xvector<xstring> vec1;
        vec1 << "one";
        vec1 << "two";
        vec1 << "three";
        vec1 << "four";
        xvector<xstring> vec1copy = vec1;
        Print(vec1.Join(" "));

        xvector<xstring> vec2{ "five", "six", "seven", "eight" };
        xvector<xstring> vec2copy = vec2;

        vec1 += vec2;

        Print(vec1.Join(" "));

        Reset(vec1copy, vec2copy, vec1, vec2);

        xvector<xstring> joined_vec = vec1 + vec2;
        Print(joined_vec.Join(" "));

        xvector<int> vec{ 0,1,2,3,4,5,6,7,8,9,10 };
        Print(vec(3, 9, 2).Join(' ')); // stream join
        // start at the 3rd element, end at the 9th element, skipping every other element

        xstring nums = joined_vec.Join(' ');
        Print(xstring("nums = ") + nums.Split(R"(\s)").Join('*'));
        cout << '\n';


        cout << "Match insensitive case: " << xstring("Ryan").Match("rYaN", rxm::icase) << endl;
        cout << "Match sensitive   case: " << xstring("Ryan").Match("rYaN") << endl;

        xstring str = "12345";
        cout << ("123" == str(0, 3)) << endl;
        cout << ("123" == str(0, -2)) << endl; // start to end (minus last 2 vars)
        cout << ("543" == str(-1, -4, -1)) << endl; // reversed (-z); x starts at the end nad goes back 3 spaces (4 - 1 = 3)
        cout << ("543" == str(4, 1, -1)) << endl; // reversed (-z); x starts at the end nad goes back 3 spaces (4 - 1 = 3)
        cout << ("321" == str(-3, -6, -1)) << endl;
        cout << ("123" == str(-6, -2, 1)) << endl; // sub 6 from (end()) to sub 2 from (end())
        cout << ("321" == str(2, -6, -1)) << endl; // sub 6 from (end()) to sub 2 from (end())

        cout << str(-6, -2, 1) << endl;

        return 0;
    }


    inline void add_n_join() {
        xstring str = "one";
        const char* two = " two ";
        str += two;
        const char* three = " three ";
        str = str + three;

        const char* four = " four ";
        for (const char* four_cp = four; four_cp != &four[strlen(four)]; four_cp++)
            str += *four_cp;


        const char* five = " five ";
        for (const char* five_cp = five; five_cp != &five[strlen(five)]; five_cp++)
            str = str + *five_cp;

        str.Print(2);
        // ---------------------------------------------------------------------
        xstring xstr = "aaa ";
        xstr += string(" bbb ");
        xstr = xstr + string(" ccc ");
        xstr += xstring(" ddd ");
        xstr = xstr + xstring(" eee ");
        xstr.Print();
    }

};

