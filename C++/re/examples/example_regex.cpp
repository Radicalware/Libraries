#include<iostream>
#include<vector>
#include<string>

#include "./re.h" // Found on "https://github.com/Radicalware"

// // g++ -g $file.cpp -o $file -std=c++17 -Wfatal-errors

using std::cout;
using std::endl;
using std::string;
using std::vector;

// ===================================================================================================

void split(){
    cout << "\n===(SPLIT)======================================================\n";
    string tester = "This is our test string!";
    vector<string> ar_tester = re::split("\\s",tester);
    for(string& i: ar_tester)
        cout << i << " |<>| ";
    // OUTPUT
    // This |<>| is |<>| our |<>| test |<>| string! |<>| 

    cout << '\n';   
}

// ===================================================================================================

void findall(){
    const char *emails = R"emails(my_email@gmail.com
subdomain.another_mailing_address@hotmail.com
Hacker.Havoc.ctr@street.district.city ctrFx
vice.crusade.ctr@us.underground.nil   ctrEx
)emails";


    cout << "\n===(FINDALL)====================================================\n";

    // Findall iterates over every line individually
    // It does the same thing as search but breaks down each line into its
    // own array element before regex and adds all its findings to a vector

    // similar to the following python command
    // re.findall(re.compile(r'regex',re.M),string_to_search)

    vector<string> findall_array = re::findall("^.*(?:\\@)(.*(\\.?).*?)(?=\\.)(.+?)$",emails);
    // vector<string> findall_array = re::findall("^.*(?:\\@)(.*(\\.?).*?)(?=\\.)(.+?)$",emails, true);

    // This regex will find the domain names with the top-level domain name stripped
    
    // If groups are set to "true" you will get all capture groups (no smart auto picking)
    // I left it commented out because it will overload the amount of data output

    for(string& i: findall_array)
        cout << "Findall_1 = " << i << endl;
              // OUTPUT
              // Findall = gmail
              // Findall = hotmail
              // Findall = nsoc.health
              // Findall = us.army

    // returns the domains with the top-level domain stripped
    cout << endl;

    findall_array = re::findall("^([^\\s]*)((\\s|$)?)",emails);
    for(string& i: findall_array)
        cout << "Findall_2 = " << i << endl;

    // Findall_2 is interesting because unlike the first one that doesn't have
    // a beginning capture group, this one dones, and we still get our the
    // correct regex sent to our vector
}
// ===================================================================================================

void search()
{
    const char *string_to_search = R"string_to_search(VVVVsentencePPPPP
AAAAsentenceBBBBB
XXXXsentenceZZZZZ
VVVVsentencePPPPP
)string_to_search";

    cout << "\n===(SEARCH)=====================================================\n";

    // Search is the same thing as Findall but it looks at the whole string at
    // one time intead of each line individually. It only returns the first 
    // with bool true specified occurance of what you are searching for 
    // (unlike findall). Also, note, you will need to target the item in the array
    // if you use bool 'true'

    vector<string> findall_array = re::search("^.*V+(.+?)P",string_to_search,true);
    // note: the "t" for groups = true 
    for(string& i: findall_array)
        cout << "Search1  = " << i << endl;
              // OUTPUT
              // Search1  = VVVVsentenceP
              // Search1  = sentence

    // Notice in capture one how I only got one "sentence" back with a "^" 
    // at the start. That is because I am using re::search and not re::findall
    // re::findall loops through each line individually, search looks at the 
    // full text as a whole.
    cout << endl;
    findall_array = re::search("(sentence)",string_to_search);
    for(string& i: findall_array)
        cout << "Search2  = " << i << endl;
                // OUPUT
                // Search2  = sentence
                // Search2  = sentence
                // Search2  = sentence
                // Search2  = sentence

    // notice how in Search2, even though we use a basic string, it still needs
    // to be used in a capture group (otherwise it won't be discovered)

}
// ===================================================================================================

void match()
{

    cout << "\n===(MATCH)======================================================\n";
    const char *string_to_search = " WORD1WORD2 majic WORD3";

    if (re::match("^(\\s?)(WORD[0-9]){0,2}\\s(.+?)\\s(WORD[0-9]){0,2}$",string_to_search))
    {
        cout << "WE FOUND A MATCH!!!\n"; // this prints
    }
    else
    {
        cout << "WE DIDN'T FIND A MATCH\n"; // this does NOT print
    }
}
// ===================================================================================================

void sub()
{

    cout << "\n===(REPLACE)====================================================\n";

    const char *string_to_replace = " WORD1WORD2 majic WORD3WORD4";
    const char* char_pattern = "WORD[0-9]";
    const std::string str_pattern  = "WORD[0-9]";

    std::string replaced = re::sub(char_pattern,"__",string_to_replace);
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
    cout << "The string has \"" << re::char_count('\n',string_to_search) << "\" char matches\n";
}

void str_count()
{
    const char *string_to_search = R"string_to_search(VVVVsentencePPPPP
AAAAsentenceBBBBB
XXXXsentenceZZZZZ
VVVVsentencePPPPP
)string_to_search";

    cout << "\n===(STR COUNT)==================================================\n";
    cout << "The string has \"" << re::str_count("(sentence)",string_to_search) << "\" str matches\n";
}


int main(){

    // The purpose of the re namespace is to keep the syntax as close to
    // what you would use in python as possible
    

    split();      // splits the string via regex

    findall();    // returns a vector of all the regex found on a per-line basis
                  // similar to the following python command
                  // re.findall(re.compile(r'regex',re.M),string_to_search)

    search();     // returns the regex_found on an re.DOTALL basis
                  // similar to (re.findall(r'regex',string_to_search,flags=re.DOTALL))
                  // re.DOTALL will search the string as a whole (ignoring new lines)

    match();      // returns "bool" = 0/1 depending of if the string matches 
                  // (use ^.*(regex).*$) if there is only a segment your searching for

    sub();        // same as re.sub NOT object.replace in python, I left the name
                  // as "sub" and not "replace" because the input placements
                  // follow that of python's logic


    char_count(); // get the number of a specific char found in a string

    str_count();  // get the number of a specific string found in a string

    cout << '\n';
    return 0;
}

// g++ -g $file.cpp -o $file -std=c++17 -Wfatal-errors
 
