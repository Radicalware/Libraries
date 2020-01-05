#pragma once

// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>

#include "xstring.h"
#include "xvector.h"

using std::cout;
using std::endl;

struct Option
{

    // NOTE: All test functions are inline to make example reading easier.

    // ===================================================================================================

    inline void split() {
        cout << "\n===(SPLIT)======================================================\n";
        xstring tester = "This is our test string!\nline one\nline two\nline three   \n\n\n  ";
        cout << (tester.trim());
        cout << "\n*****" << endl;

        // IMPORTANT!! Do not use '\s' for a space, use a literal space in the example below
        cout << tester.split(' ').join("--") << endl;

        cout << "\n*****" << endl;
        cout << tester.split('\n').join("\n >> ") << "\n\n";

        // OUTPUT
        // This |<>| is |<>| our |<>| test |<>| string! |<>| 
    }

    // ===================================================================================================

    inline void findall() {
        xstring emails = R"emails(my_email@gmail.com garbage
un-needed subdomain.another_mailing_address@hotmail.com
Hacker.Havoc.ctr@street.district.city junktext
more junk text vice.crusade.ctr@us.underground.nil   junk
)emails";

        
        cout << "\n===(FINDWALK/FINDALL)===========================================\n";

        // findwalk = regex over every element line-by-line
        // findall  = regex over everything at once

        xvector<xstring> finder;
        xstring header;

        header = "\nFindall 1 = "; 
        header.print(0);
        emails.findwalk(R"(([^\s]*\@[^\s]*))").join(header).print(2);
        // return the whole email
        // first  '(' is for string literal indicated by the 'R'
        // second '(' is for our primary capture group (anything outside will be omitted) from the capture returned

        // ------------------- OUTPUT BELOW --------------------------
        //Findall 1 = my_email@gmail.com
        //Findall 1 = subdomain.another_mailing_address@hotmail.com
        //Findall 1 = Hacker.Havoc.ctr@street.district.city
        //Findall 1 = vice.crusade.ctr@us.underground.nil
        // -----------------------------------------------------------

        header = "\nFindall 2 = "; 
        header.print(0);
        emails.findwalk(R"rex((?:.*\@)(.+(\.?).+?)(?=\.))rex").join(header).print(2);

        // a big advantage here is that look-ahead/behind is not a finite length like in python
        // lookahead  = (?:<regex>)
        // lookbehind = (?=<regex>)

        // returns the domains with the top-level domain stripped
        // ----- OUTPUT BELOW --------------
        // Findall 2 = gmail
        // Findall 2 = hotmail
        // Findall 2 = street.district
        // Findall 2 = us.underground
        // ---------------------------------

        // next we do the same thing but regex over the whole thing at once
        header = "\nFindall 3 = "; 
        header.print(0);
        emails.findall(R"rex((?:(?:^|\n).*\@)(.+(\.?).+?)(?=\.))rex").join(header).print(2);

        // ----- OUTPUT BELOW --------------
        // Findall 2 = gmail
        // Findall 2 = hotmail
        // Findall 2 = street.district
        // Findall 2 = us.underground
        // ---------------------------------
    }


    inline void search() {
        xstring emails = R"email(more junk text vice.crusade.ctr@us.underground.nil   junk)email";

        cout << "Found capture groups\n";
        emails.search(R"((vice).(crusade).(ctr)@(us).(underground))", 5).join('\n').print(2);
        // 5 idicates we want the first 5 on (opposed to off) capture groups

    }
    // ===================================================================================================

    inline void match()
    {

        cout << "\n===(MATCH)======================================================\n";
        xstring string_to_search = " WORD1WORD2 majic WORD3";

        if (string_to_search.match("^(\\s?)(WORD[0-9]){0,2}\\s(.+?)\\s(WORD[0-9]){0,2}$"))
        {
            cout << "WE FOUND A MATCH!!!\n"; // this prints
        }
        else
        {
            cout << "WE DIDN'T FIND A MATCH\n"; // this does NOT print
        }

        xstring emails = R"(my_email@gmail.com
subdomain.another_mailing_address@hotmail.com
Hacker.Havoc.ctr@street.district.city 
vice.crusade.ctr@us.underground.nil
)";
        // if you need to feed in a vector split at regex spot you can either
        // feed in an re::split or use ac::match_one // ac::match_all

        cout << "match all lines = " << emails.match_lines("^.*(?:\\@)(GMAIL)\\.(com)$", rxm::icase) << endl;
        // returns false since all the lines don't contain gmail

        cout << "match one line  = " << emails.match_line("^.*(?:\\@)(GMAIL)\\.(com)$", rxm::icase) << endl;
        // returns true sense at lease one line contains gmail

        cout << "match all lines = " << emails.match_lines("^.*(@).*$") << endl;
        // returns true sense all lines have a "@"
    }

    // ===================================================================================================

    inline void sub()
    {

        cout << "\n===(REPLACE)====================================================\n";

        xstring string_to_replace = " WORD1WORD2 majic WORD3WORD4";

        cout << string_to_replace.sub("(WORD[0-9])", "__") << endl;
        // ____ majic ____
    }


    inline void char_count()
    {
        xstring string_to_search = R"string_to_search(VVVVsentencePPPPP
AAAAsentenceBBBBB
XXXXsentenceZZZZZ
VVVVsentencePPPPP
)string_to_search";

        cout << "\n===(CHAR COUNT)=================================================\n";
        cout << "The string has \"" << string_to_search.count('\n') << "\" char matches\n";
    }

    inline void str_count()
    {
        xstring string_to_search = R"string_to_search(VVVVsentencePPPPP
AAAAsentenceBBBBB
XXXXsentenceZZZZZ
VVVVsentencePPPPP
)string_to_search";

        cout << "\n===(STR COUNT)==================================================\n";
        cout << "The string has \"" << string_to_search.count("(sentence)") << "\" str matches\n";
    }

};

