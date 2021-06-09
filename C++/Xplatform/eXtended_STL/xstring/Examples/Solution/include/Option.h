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

    inline void Split() {
        cout << "\n===(SPLIT)======================================================\n";
        xstring tester = "This is our test string!\nline one\nline two\nline three   \n\n\n  ";
        cout << (tester.Trim());
        cout << "\n*****" << endl;

        // IMPORTANT!! Do not use '\s' for a space, use a literal space in the example below
        cout << tester.Split(' ').Join("--") << endl;

        cout << "\n*****" << endl;
        cout << tester.Split('\n').Join("\n >> ") << "\n\n";

        // OUTPUT
        // This |<>| is |<>| our |<>| test |<>| string! |<>| 
    }

    // ===================================================================================================

    inline void Findall() {
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

        cout << "------------ FIND WALK FULL EMAIL -----------------------------\n";
        header = "\nFindall 1 = "; 
        header.Print(0);
        emails.Findwalk(R"(([^\s]*\@[^\s]*))").Join(header).Print(2);
        cout << "----\n";
        header.Print(0);
        emails.Findwalk(RE2(R"(([^\s]*\@[^\s]*))")).Join(header).Print(2);

        // return the whole email
        // first  '(' is for string literal indicated by the 'R'
        // second '(' is for our primary capture group (anything outside will be omitted) from the capture returned

        // ------------------- OUTPUT BELOW --------------------------
        //Findall 1 = my_email@gmail.com
        //Findall 1 = subdomain.another_mailing_address@hotmail.com
        //Findall 1 = Hacker.Havoc.ctr@street.district.city
        //Findall 1 = vice.crusade.ctr@us.underground.nil
        // -----------------------------------------------------------

        cout << "------------ FIND WALK EMAIL DOMAIN ---------------------------\n";
        header = "\nFindall 2 = "; 
        header.Print(0);
        emails.Findwalk(R"rex((?:.*\@)(.+(\.?).+?)(?:\.))rex").Join(header).Print(2);
        cout << "----\n";
        header.Print(0);
        emails.Findwalk(RE2(R"rex((?:.*\@)(.+(\.?).+?)(?:\.))rex")).Join(header).Print(2);

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
        cout << "------------ FIND ALL EMAIL DOMAIN ----------------------------\n";
        header = "\nFindall 3 = ";
        header.Print(0);
        emails.Findall(R"rex((?:(?:^|\n).*\@)(.+(\.?).+?)(?:\.))rex").Join(header).Print(2);
        cout << "----\n";
        header.Print(0);
        emails.Findall(RE2(R"rex((?:(?:^|\n).*\@)(.+(\.?).+?)(?:\.))rex")).Join(header).Print(2);

        // ----- OUTPUT BELOW --------------
        // Findall 2 = gmail
        // Findall 2 = hotmail
        // Findall 2 = street.district
        // Findall 2 = us.underground
        // ---------------------------------
    }


    inline void Search() {
        xstring emails = R"email(more junk text vice.crusade.ctr@us.underground.nil   junk)email";

        cout << "Found capture groups\n";
        emails.Search(R"((vice).(crusade).(ctr)@(us).(underground))", rxm::ECMAScript, 5).Join("==").Print(2);
        // 5 idicates we want the first 5 on (opposed to off) capture groups
    }
    // ===================================================================================================

    inline void Match()
    {

        cout << "\n===(MATCH)======================================================\n";
        xstring string_to_search = " WORD1WORD2 majic WORD3";

        if (string_to_search.Match("^(\\s?)(WORD[0-9]){0,2}\\s(.+?)\\s(WORD[0-9]){0,2}$"))
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
        // feed in an RE::split or use AC::match_one // AC::match_all

        cout << "match all lines = " << emails.MatchAllLines("^.*(?:\\@)(GMAIL)\\.(com)$", rxm::icase) << endl;
        // returns false since all the lines don't contain gmail

        cout << "match one line  = " << emails.MatchAllLines("^.*(?:\\@)(GMAIL)\\.(com)$", rxm::icase) << endl;
        // returns true sense at lease one line contains gmail

        cout << "match all lines = " << emails.MatchAllLines("^.*(@).*$") << endl;
        // returns true sense all lines have a "@"
    }

    // ===================================================================================================

    inline void Sub()
    {

        cout << "\n===(REPLACE)====================================================\n";

        xstring string_to_replace = " WORD1WORD2 majic WORD3WORD4";

        cout << string_to_replace.Sub("(WORD[0-9])", "__") << endl;
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
        cout << "The string has \"" << string_to_search.Count('\n') << "\" char matches\n";
    }

    inline void str_count()
    {
        xstring string_to_search = R"string_to_search(
            VVVVsentencePPPPP 
            AAAAsentenceBBBBB
            XXXXsentenceZZZZZ
            VVVVsentencePPPPP )string_to_search";

        cout << "\n===(STR COUNT)==================================================\n";
        cout << "The string has \"" << string_to_search.Count("sentence") << "\" str matches\n";
    }
};

