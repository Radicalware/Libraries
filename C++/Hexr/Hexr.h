/*
* Copyright[2018][Joel Leagues aka Scourge]
* Scourge /at\ protonmail /dot\ com
* www.Radicalware.com
* https://www.youtube.com/channel/UCivwmYxoOdDT3GmDnD0CfQA/playlists
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http ://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#ifndef _Hexr_
#define _Hexr_


#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <regex>    // Sort between ascii_string (as) and hex_string (hs)
#include <stdio.h>  // uint64_t

// g++ fuz-hexr.cpp Hexr.hpp -std=c++17 -Wfatal-errors -o ex && ./ex

typedef std::vector<std::string> sVector;
typedef std::vector<int>         iVector;
typedef std::vector<char>        cVector;

using std::string;
using std::cout;
using std::vector;


// results from this uri are shown in the following
// https://stackoverflow.com/questions/3664272/is-stdvector-so-much-slower-than-plain-arrays

// UseArray completed in 2.619 seconds
// UseVector completed in 9.284 seconds
// UseVectorPushBack completed in 14.669 seconds
// The whole thing completed in 26.591 seconds

// That is why I use arrays here and not vectors by default

class Hexr
{
private:
	enum e_start_type
	{
		// Order Based on Path1

		start_ascii_string,

		start_char_array,
		start_int_array,
		start_hex_array,

		start_hex_string,
	};

	enum e_ret_type // used to enhance speed when making comparisons
	{               // the alternative would be comparing the strings from input which is slow

		as, // ascii_string

		ca, // char_array
		ia, // int_array
		ha, // hex_array

		hs,  // hex_string

		none,// occurs when programmer asks for a full instnace
	};

	struct st_converted
	{
		bool out_as;
		bool out_ca;
		bool out_ia;
		bool out_ha;
		bool out_hs;
	};

	e_start_type  m_start_type;
	e_ret_type    r_ret_type; // r_ means that this var is used in conditionals to decide what is returned.
	st_converted  m_converted{ false,false,false,false,false };

	bool m_vector_in_char_array = false;

	string   s_string; // handling of the string input before it gets added to the conversion loop

	char    *s_char_array = new char[1]{};
	int     *s_int_array = new int[1]{};
	string  *s_hex_array = new string[1]{};

	int      m_old_value = 0;

	string   m_insert = " ";   // optional: if an array is converted to a string, insert gets added between the elements
	string   m_ret_type = "";  // optional: prevents a whole loop of conversions to enhance speed
	uint64_t m_byte_count = 0;

	string   m_asm_string = "";  // for converting an array/vector to string format

	bool     m_fast_track = true;// Skip conversions to enhance speed (Don't use if you make an instance)

	char    *m_char_array = new char[1]{};
	int     *m_int_array = new int[1]{};
	string  *m_hex_array = new string[1];

	cVector  s_char_vector;
	iVector  s_int_vector;
	sVector  s_hex_vector;

	cVector  m_char_vector;
	iVector  m_int_vector;
	sVector  m_hex_vector;

	string   m_ascii_string;
	string   m_hex_string;

	string   m_update_insert = "";

	uint64_t m_man_int = 0;

	bool     m_vector_used = false; // used to decide if memory deletion should be used

public:

	// =======================================================================================================
	// Path1: ascii_string > char_array > int_array > hex_array > hex_string
	void ascii_string_2_char_array();
	void char_array_2_int_array();
	void int_array_2_hex_array();
	void hex_array_2_hex_string();
	// =======================================================================================================
	// Path2: hex_string > hex_array > int_array > char_array > ascii_string
	void hex_string_2_hex_array();
	void hex_array_2_int_array();
	void int_array_2_char_array();
	void char_array_2_ascii_string();
	// =======================================================================================================

	// =======================================================================================================
	// Miscellaneous Functions

	// Modifiers
	void     trim_asm_stirng();
	void     set_int_array(int *s_int_array);
	void     set_old_value(int p_value);
	void     set_byte_count(uint64_t p_byte_cout);
	void     update_insert(string& s_update_insert);
	void     bytes(uint64_t s_bytes);
	void     manual_bytes();

	// Standard Getters
	string   ret_type();
	string   str();        // more readable
	string   s();          // faster to type	
	uint64_t byte_count(); // more readable
	uint64_t bc();         // faster to type

						   // Targeted Getters
	string   ascii_string();  // 1 ascii_string
	char    *char_array();    // 2 char_array
	cVector  char_vector();
	int     *int_array();	  // 3 int_array
	iVector  int_vector();
	string  *hex_array();     // 4 hex_array
	sVector  hex_vector();
	string   hex_string();    // 5 = hex_string

							  // debugging
	void     dbg() { cout << "------- hit -------\n"; }
	void     c(string i) { cout << "debug = " << i << '\n'; }
	void     c(int    i) { cout << "debug = " << i << '\n'; }
	// =======================================================================================================


	// =======================================================================================================
	// Initializationa and Destruction

	// If you can make the following operate with forward declaration and have the
	// default inputs, please send a request with your solution, Thanks
	void init_strings();
	Hexr(string& tmp_string, string tmp_ret_type = "", string tmp_insert = " ") :
		s_string(tmp_string), m_ret_type(tmp_ret_type), m_insert(tmp_insert)
	{
		init_strings();
	};
	Hexr(string& tmp_string, uint64_t tmp_man_int, string tmp_ret_type = "", string tmp_insert = " ") :
		s_string(tmp_string), m_man_int(tmp_man_int), m_ret_type(tmp_ret_type), m_insert(tmp_insert)
	{
		init_strings();
	};
	// -------------------------------------------------------------------------------------------------------
	// Char Array
	void init_char_array();
	void init_char_array_bytes();
	Hexr(char* tmp_char_array, string tmp_ret_type = "", string tmp_insert = " ") :
		s_char_array(tmp_char_array), m_ret_type(tmp_ret_type), m_insert(tmp_insert)
	{
		init_char_array_bytes();
	}
	Hexr(char* tmp_char_array, uint64_t tmp_man_int, string tmp_ret_type = "", string tmp_insert = " ") :
		s_char_array(tmp_char_array), m_man_int(tmp_man_int), m_ret_type(tmp_ret_type), m_insert(tmp_insert)
	{
		init_char_array_bytes();
	}

	void init_char_vector_bytes();
	Hexr(cVector& tmp_char_vector, string tmp_ret_type = "", string tmp_insert = " ") :
		s_char_vector(tmp_char_vector), m_ret_type(tmp_ret_type), m_insert(tmp_insert)
	{
		init_char_vector_bytes();
	}
	Hexr(cVector& tmp_char_vector, uint64_t tmp_man_int, string tmp_ret_type = "", string tmp_insert = " ") :
		s_char_vector(tmp_char_vector), m_man_int(tmp_man_int), m_ret_type(tmp_ret_type), m_insert(tmp_insert)
	{
		init_char_vector_bytes();
	}
	// -------------------------------------------------------------------------------------------------------
	// Int Array
	void init_int_array();
	void init_int_array_bytes();
	Hexr(int* tmp_int_array, string tmp_ret_type = "", string tmp_insert = " ") :
		s_int_array(tmp_int_array), m_ret_type(tmp_ret_type), m_insert(tmp_insert)
	{
		init_int_array_bytes();
	}
	Hexr(int* tmp_int_array, uint64_t tmp_man_int, string tmp_ret_type = "", string tmp_insert = " ") :
		s_int_array(tmp_int_array), m_man_int(tmp_man_int), m_ret_type(tmp_ret_type), m_insert(tmp_insert)
	{
		init_int_array_bytes();
	}

	void init_int_vector_bytes();
	Hexr(iVector& tmp_int_vector, string tmp_ret_type = "", string tmp_insert = " ") :
		s_int_vector(tmp_int_vector), m_ret_type(tmp_ret_type), m_insert(tmp_insert)
	{
		init_int_vector_bytes();
	}
	Hexr(iVector& tmp_int_vector, uint64_t tmp_man_int, string tmp_ret_type = "", string tmp_insert = " ") :
		s_int_vector(tmp_int_vector), m_man_int(tmp_man_int), m_ret_type(tmp_ret_type), m_insert(tmp_insert)
	{
		init_int_vector_bytes();
	}
	// -------------------------------------------------------------------------------------------------------
	// Hex Array
	void init_hex_array();
	Hexr(string* tmp_hex_array, string tmp_ret_type = "", string tmp_insert = " ") :
		s_hex_array(tmp_hex_array), m_ret_type(tmp_ret_type), m_insert(tmp_insert)
	{
		init_hex_array();
	}
	Hexr(string* tmp_hex_array, uint64_t tmp_man_int, string tmp_ret_type = "", string tmp_insert = " ") :
		s_hex_array(tmp_hex_array), m_man_int(tmp_man_int), m_ret_type(tmp_ret_type), m_insert(tmp_insert)
	{
		init_hex_array();
	}

	void init_hex_vector();
	Hexr(sVector& tmp_hex_vector, string tmp_ret_type = "", string tmp_insert = " ") :
		s_hex_vector(tmp_hex_vector), m_ret_type(tmp_ret_type), m_insert(tmp_insert)
	{
		init_hex_vector();
	}
	Hexr(sVector& tmp_hex_vector, uint64_t tmp_man_int, string tmp_ret_type = "", string tmp_insert = " ") :
		s_hex_vector(tmp_hex_vector), m_man_int(tmp_man_int), m_ret_type(tmp_ret_type), m_insert(tmp_insert)
	{
		init_hex_vector();
	}
	// -------------------------------------------------------------------------------------------------------
	// Misc starters and finishers
	Hexr();
	int get_path();
	void id_ret_type();
	bool done();
	~Hexr();
	// =======================================================================================================
};

#endif 
 
