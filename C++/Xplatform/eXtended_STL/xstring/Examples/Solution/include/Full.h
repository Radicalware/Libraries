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
    inline void print(T val) {
        cout << val << endl;
    }

    inline void reset(const xvector<xstring>& saved1, const xvector<xstring>& saved2,
        xvector<xstring>& throwback1, xvector<xstring>& throwback2)
    {
        throwback1 = saved1;
        throwback2 = saved2;
    }

    inline int Basics() {
        xstring hello_world("Hello World");
        print(hello_world.split(' ').join(" ** "));

        print(xstring("Result 'World' Found: ") + to_xstring(hello_world.scan("[wW][0oO]rld$")));
        print(xstring("Result 'World' Found: ") + to_xstring(hello_world.match("^.*[wW][0oO]rld$")));

        xvector<xstring> vec1;
        vec1 << "one";
        vec1 << "two";
        vec1 << "three";
        vec1 << "four";
        xvector<xstring> vec1copy = vec1;
        print(vec1.join(" "));

        xvector<xstring> vec2{ "five", "six", "seven", "eight" };
        xvector<xstring> vec2copy = vec2;

        vec1 += vec2;

        print(vec1.join(" "));

        reset(vec1copy, vec2copy, vec1, vec2);

        xvector<xstring> joined_vec = vec1 + vec2;
        print(joined_vec.join(" "));

        xvector<int> vec{ 0,1,2,3,4,5,6,7,8,9,10 };
        print(vec(3, 9, 2).join(' ')); // stream join
        // start at the 3rd element, end at the 9th element, skipping every other element

        xstring nums = joined_vec.join(' ');
        print(xstring("nums = ") + nums.split(R"(\s)").join('*'));
        cout << '\n';


        cout << "Match insensitive case: " << xstring("Ryan").match("rYaN", rxm::icase) << endl;
        cout << "Match sensitive   case: " << xstring("Ryan").match("rYaN") << endl;

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

        str.print(2);
        // ---------------------------------------------------------------------
        xstring xstr = "aaa ";
        xstr += string(" bbb ");
        xstr = xstr + string(" ccc ");
        xstr += xstring(" ddd ");
        xstr = xstr + xstring(" eee ");
        xstr.print();
    }

};

