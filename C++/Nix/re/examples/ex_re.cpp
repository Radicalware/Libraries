#include<iostream>
#include<vector>
#include<string>

#include "re.h"
#include "ac.h"


using std::cout;
using std::endl;
using std::string;
using std::vector;

// ===================================================================================================

void split() {
	cout << "\n===(SPLIT)======================================================\n";
	string tester = "This is our test string!\nline one\nline two\nline three   \n\n\n  ";

    cout << re::strip(tester) << endl;
	vector<string> ar_tester = re::split("\\s", re::strip(tester));
    cout << ac::join(ar_tester, "--");

    cout << "\n*****\n";
    ar_tester = re::split('\n', re::strip(tester));
    cout << ac::join(ar_tester, "\n >> ");


	// OUTPUT
	// This |<>| is |<>| our |<>| test |<>| string! |<>|

	cout << '\n';
}

// ===================================================================================================

void findall() {
	const char *emails = R"emails(my_email@gmail.com
subdomain.another_mailing_address@hotmail.com
Hacker.Havoc.ctr@street.district.city ctrFx
vice.crusade.ctr@us.underground.nil   ctrEx
)emails";


	cout << "\n===(FINDALL)====================================================\n";

	// Findall iterates over every line individually

	vector<string> findall_array = re::findall(R"(^(?:(.+\@))(.+(\.?).+?)(?=\.))", emails);
	// return only the capturegroup of interest

	//vector<string> findall_array = re::findall(R"(^(?:(.+\@))(.+(\.?).+?)(?=\.))", emails, true);
	// returns the full capture group, look behind, and look ahead seperatly

	for (string& i : findall_array)
		cout << "Findall_1 = " << i << endl;
	// OUTPUT
	// Findall = gmail
	// Findall = hotmail
	// Findall = nsoc.health
	// Findall = us.army

	// returns the domains with the top-level domain stripped
	cout << endl;

	findall_array = re::findall("^([^\\s]*)((\\s|$)?)", emails);
	// return the whole email
	for (string& i : findall_array)
		cout << "Findall_2 = " << i << endl;

	// Findall_2 is interesting because unlike the first one that doesn't have
	// a beginning capture group, this one dones, and we still get our the
	// correct regex sent to our vector
}

// ===================================================================================================

void match()
{

	cout << "\n===(MATCH)======================================================\n";
	const char *string_to_search = " WORD1WORD2 majic WORD3";

	if (re::match("^(\\s?)(WORD[0-9]){0,2}\\s(.+?)\\s(WORD[0-9]){0,2}$", string_to_search))
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
	// feed in an re::split or use ac::match_one // ac::match_all

	cout << "match lines 'all' = " << re::match_lines("^.*(?:\\@)(gmail)\\.(com)$", emails) << endl;
	// all returns falls sense all the lines don't contain gmail

	cout << "match lines 'one' = " << re::match_line("^.*(?:\\@)(gmail)\\.(com)$", emails) << endl;
	// returns true sense at lease one line contains gmail

	cout << "match lines 'all' = " << re::match_lines("^.*(@).*$", emails) << endl;
	// true sense all lines have a "@"
}

// ===================================================================================================

void sub()
{

	cout << "\n===(REPLACE)====================================================\n";

	const char *string_to_replace = " WORD1WORD2 majic WORD3WORD4";
	const char* char_pattern = "WORD[0-9]";
	const std::string str_pattern = "WORD[0-9]";

	std::string replaced = re::sub(char_pattern, "__", string_to_replace);
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
	cout << "The string has \"" << re::count('\n', string_to_search) << "\" char matches\n";
}

void str_count()
{
	const char *string_to_search = R"string_to_search(VVVVsentencePPPPP
AAAAsentenceBBBBB
XXXXsentenceZZZZZ
VVVVsentencePPPPP
)string_to_search";

	cout << "\n===(STR COUNT)==================================================\n";
	cout << "The string has \"" << re::count("(sentence)", string_to_search) << "\" str matches\n";
}

void search() { /* This is the only function that doesn't exist yet */ }


int main()
{

	// The purpose of the re namespace is to keep the syntax as close to
	// what you would use in python as possible


	split();      // splits the string via regex

	findall();    // returns a vector of all the regex found on a per-line basis
				  // similar to the following python command
				  // re.findall(re.compile(r'regex',re.M),string_to_search)

	match();      // returns "bool" = 0/1 depending of if the string matches 
				  // (use ^.*(regex).*$) if there is only a segment your searching for

	sub();        // same as re.sub NOT object.replace in python, I left the name
				  // as "sub" and not "replace" because the input placements
				  // follow that of python's logic

	char_count(); // get the number of a specific char found in a string

	str_count();  // get the number of a specific string found in a string
	
	search();     // This is the only one that does not exist yet.
				  // Later on, if I find it a priority I will make a search object
				  // That will house an array of groups matched. 

	cout << '\n';
	return 0;

}
