#pragma once


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
		cout << tester.strip();
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
		xstring emails = R"emails(my_email@gmail.com
subdomain.another_mailing_address@hotmail.com
Hacker.Havoc.ctr@street.district.city ctrFx
vice.crusade.ctr@us.underground.nil   ctrEx
)emails";


		cout << "\n===(FINDALL)====================================================\n";

		// Findall iterates over every line individually

		xvector<xstring> findall_array = emails.findall(R"rex(^(?:.*\@)(.+(\.?).+?)(?=\.))rex");
		// return only the capturegroup of interest

		//vector<string> findall_array = re::findall(R"(^(?:(.+\@))(.+(\.?).+?)(?=\.))", emails, true);
		// returns the full capture group, look behind, and look ahead seperatly

		cout << "Findall_1 = " << findall_array.join("\nFindall_1 = ") << endl;
		// OUTPUT
		// Findall = gmail
		// Findall = hotmail
		// Findall = nsoc.health
		// Findall = us.army

		// returns the domains with the top-level domain stripped

		findall_array = emails.findall("^([^\\s]*)((\\s|$)?)");
		// return the whole email
		cout << findall_array.join("Findall_2 = ") << endl;

		// Findall_2 is interesting because unlike the first one that doesn't have
		// a beginning capture group, this one dones, and we still get our the
		// correct regex sent to our vector

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

		cout << "match lines 'all' = " << emails.match_lines("^.*(?:\\@)(gmail)\\.(com)$") << endl;
		// all returns falls sense all the lines don't contain gmail

		cout << "match lines 'one' = " << emails.match_line("^.*(?:\\@)(gmail)\\.(com)$") << endl;
		// returns true sense at lease one line contains gmail

		cout << "match lines 'all' = " << emails.match_lines("^.*(@).*$") << endl;
		// true sense all lines have a "@"
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

	inline void search() { 
		/* 
		This is the only function that doesn't exist yet.
		I plan to make Search its own object at a later time.
		*/ 
	}

};

