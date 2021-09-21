
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>
#include<vector>
#include<string>

#include "RE.h"
#include "AC.h"


using std::cout;
using std::endl;
using std::string;
using std::vector;

// ===================================================================================================

void Split() {
    cout << "\n===(SPLIT)======================================================\n";
    string tester = "This is our test string!\nline one\nline two\nline three   \n\n\n  ";

    cout << RE::Strip(tester);
    cout << "\n*****" << endl;

    vector<string> ar_tester = RE::Split("\\s", RE::Strip(tester));
    cout << AC::Join(ar_tester, "--");

    cout << "\n*****" << endl;
    ar_tester = RE::Split('\n', RE::Strip(tester));
    cout << AC::Join(ar_tester, "\n >> ");


    // OUTPUT
    // This |<>| is |<>| our |<>| test |<>| string! |<>| 

    cout << '\n';
}

// ===================================================================================================

void Findall() {


        std::string emails = R"emails(my_email@gmail.com garbage
un-needed subdomain.another_mailing_address@hotmail.com
Hacker.Havoc.ctr@street.district.city junktext
more junk text vice.crusade.ctr@us.underground.nil   junk
)emails";

    cout << "\n===(FINDALL)====================================================\n";

    // Findall iterates over every line individually

    vector<string> findall_array = RE::Findall(R"rex((?:(?:^|\n).*\@)(.+(\.?).+?)(?=\.))rex", emails);
    // findall uses regex all at once (notice tha twe use nested look-aheads to account for this)

    for (string& i : findall_array)
        cout << "Findall_1 = " << i << endl;
    // OUTPUT
    // Findall_1 = gmail
    // Findall_1 = hotmail
    // Findall_1 = street.district
    // Findall_1 = us.underground

    cout << endl;

    findall_array = RE::Findwalk(R"(([^\s]*\@[^\s]*))", emails); // findwalk uses regex line/by/line

    for (string& i : findall_array)
        cout << "Findall_2 = " << i << endl;

    // OUTPUT
    // Findall_2 = my_email@gmail.com
    // Findall_2 = subdomain.another_mailing_address@hotmail.com
    // Findall_2 = Hacker.Havoc.ctr@street.district.city
    // Findall_2 = vice.crusade.ctr@us.underground.nil
}

// ===================================================================================================

void Match()
{

    cout << "\n===(MATCH)======================================================\n";
    const char *string_to_search = " WORD1WORD2 majic WORD3";

    if (RE::Match("^(\\s?)(WORD[0-9]){0,2}\\s(.+?)\\s(WORD[0-9]){0,2}$", string_to_search))
    {
        cout << "WE FOUND A MATCH!!!\n"; // this prints
    }
    else
    {
        cout << "WE DIDN'T FIND A MATCH\n"; // this does NOT print
    }

    std::string emails = R"(my_email@gmail.com
subdomain.another_mailing_address@hotmail.com
Hacker.Havoc.ctr@street.district.city 
vice.crusade.ctr@us.underground.nil
)";
    // if you need to feed in a vector split at regex spot you can either
    // feed in an RE::split or use AC::match_one // AC::match_all

    cout << "match lines 'all' = " << RE::MatchAllLines("^.*(?:\\@)(gmail)\\.(com)$", emails) << endl;
    // all returns falls sense all the lines don't contain gmail

    cout << "match lines 'one' = " << RE::MatchLine("^.*(?:\\@)(gmail)\\.(com)$", emails) << endl;
    // returns true sense at lease one line contains gmail

    cout << "match lines 'all' = " << RE::MatchAllLines("^.*(@).*$", emails) << endl;
    // true sense all lines have a "@"
}

// ===================================================================================================

void Sub()
{

    cout << "\n===(REPLACE)====================================================\n";

    const char *string_to_replace = " WORD1WORD2 majic WORD3WORD4";
    const char* char_pattern = "WORD[0-9]";
    const std::string str_pattern = "WORD[0-9]";

    std::string replaced = RE::Sub(char_pattern, "__", string_to_replace);
    cout << replaced << endl; // ____ majic ____
}


void char_count()
{
    const char *string_to_search = R"string_to_search(VVVVsentencePPPPP
AAAAsentenceBBBBB
XXXXsentenceZZZZZ
VVVVsentencePPPPP
)string_to_search";

    cout << "\n===(CHAR COUNT)=================================================\n";
    cout << "The string has \"" << RE::Count('\n', string_to_search) << "\" char matches\n";
}

void str_count()
{
    const char *string_to_search = R"string_to_search(VVVVsentencePPPPP
AAAAsentenceBBBBB
XXXXsentenceZZZZZ
VVVVsentencePPPPP
)string_to_search";

    cout << "\n===(STR COUNT)==================================================\n";
    cout << "The string has \"" << RE::Count("(sentence)", string_to_search) << "\" str matches\n";
}



int main()
{
    // The purpose of the re namespace is to keep the syntax as close to
    // what you would use in python as possible


    Split();      // splits the string via regex

    Findall();    // findwalk/findall returns a all the matches for a given regex

    Match();      // returns "bool" = 0/1 depending of if the string matches 
                  // (use ^.*(regex).*$) if there is only a segment your searching for

    Sub();        // same as re.sub NOT object.replace in python, I left the name
                  // as "sub" and not "replace" because the input placements
                  // follow that of python's logic

    char_count(); // get the number of a specific char found in a string

    str_count();  // get the number of a specific string found in a string
    


    cout << '\n';




    return 0;

}
