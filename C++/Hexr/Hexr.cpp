
/*
* Copyright[2018][Joel Leagues aka Scourge]
* Scourge /at\ protonmail /dot\ com
* www.Radicalware.com
* https://www.youtube.com/channel/UCivwmYxoOdDT3GmDnD0CfQA/playlists
*
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


//#include "stdafx.h" // <<<<< FOR VISUAL STUDIO 
//#include "stdint.h" // <<<<< FOR VISUAL STUDIO 
// If there is a way to get this to work without manually adding it, please let me know. Thanks


#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <regex>    // Sort between ascii_string (as) and hex_string (hs)
#include <stdio.h>  // uint64_t

#include "Hexr.h"


typedef std::vector<std::string> sVector;
typedef std::vector<int>         iVector;
typedef std::vector<char>        cVector;

using std::cout;
using std::endl;
using std::hex; // and my method will store the result in a (variable & hex array)
using std::stoi;
using std::ostringstream;
using std::string;
using std::vector;
using std::to_string;

using std::regex;
using std::smatch;
using std::regex_match;
using std::sregex_token_iterator;

Hexr::Hexr() {
	cout << "Hexer start with one of the following\n" <<
		"1.) Ascii String\n" <<
		"2.) Char Array/Vector\n" <<
		"3.) Int Array/Vector\n" <<
		"4.) Hex Array/Vector\n" <<
		"5.) Hex String\n" <<
		"Else Hexr will crash\n"; exit(1);
};

// overload (ascii_string & hex_string) >>>

void Hexr::init_strings() {
	// regex to differentiate between (hex_string) & (ascii_string)
	id_ret_type();
	if (r_ret_type == none) { m_fast_track = false; }
	if (bool(regex_match(s_string, regex(R"(^((0x)?)[0-9A-Fa-f]*$)"))) == true) {
		if (m_man_int == 0) { m_byte_count = s_string.length() / 2 - 1; }
		else { m_byte_count = m_man_int; }

		if (s_string[0] == '0' && s_string[1] == 'x') {
			s_string.erase(0, 2);
		}
		m_start_type = start_hex_string;
		m_hex_string = s_string;
		m_converted.out_hs = true;
		hex_string_2_hex_array();
	}
	else {
		if (m_man_int == 0) { m_byte_count = s_string.length(); }
		else { m_byte_count = m_man_int; }

		m_start_type = start_ascii_string;
		m_ascii_string = s_string;
		m_converted.out_as = true;
		ascii_string_2_char_array();
	}
};

// << strings
// overload char array >>>
void Hexr::init_char_array() {
	if (r_ret_type == none) { m_fast_track = false; }
	m_start_type = start_char_array;
	char **joined_char = &s_char_array;
	m_ascii_string = *joined_char;
	m_converted.out_as = true;

	delete[] m_char_array;
	m_char_array = s_char_array;
	m_converted.out_ca = true;

	if (r_ret_type == ia && m_converted.out_ia == true) { char_array_2_int_array(); }
	else if (r_ret_type == as) { char_array_2_ascii_string(); }
	else if (r_ret_type == ca) { ascii_string_2_char_array(); }
	else { char_array_2_int_array(); }
}


void Hexr::init_char_array_bytes() {
	id_ret_type();
	manual_bytes();

	if (m_byte_count == 0) {
		int current_byte = 0;
		while (true) {
			// debug
			current_byte = int(s_char_array[m_byte_count]);
			if (current_byte > 255 || current_byte < 1) { break; }
			m_int_vector.push_back(current_byte);
			m_byte_count += 1;
		}
		delete[] s_int_array; delete[] m_int_array;
		s_int_array = &m_int_vector[0];
		m_int_array = s_int_array;
		set_int_array(m_int_array);
		m_converted.out_ia = true;
		m_vector_in_char_array = true;
	}

	init_char_array();
}


void Hexr::init_char_vector_bytes() {
	id_ret_type();
	if (m_man_int > 0) { m_byte_count = m_man_int; }
	else { m_byte_count = s_char_vector.size(); }
	m_vector_used = true;
	delete[] s_char_array;
	s_char_array = &s_char_vector[0];

	// capture the address of the base pointer
	// the [0] is treated like a '*' in an array but always 
	// targets the first element instead of the base address,
	// which may || may not be the first element (depending on if
	// we are talking about an array || a vector)
	// ps. I use || instead of 'o r' to remove red marks in Visual Studio
	init_char_array();
}

// overload char array <<<
// overload int array >>>

void Hexr::init_int_array() {
	delete[] m_int_array;
	m_start_type = start_int_array;
	m_int_array = s_int_array;
	m_converted.out_ia = true;

	if (r_ret_type == ia) { char_array_2_int_array(); }
	else if (get_path() == 1) { int_array_2_hex_array(); }
	else { int_array_2_char_array(); }
}

void Hexr::init_int_array_bytes() {
	id_ret_type();
	manual_bytes();

	if (m_byte_count == 0) {
		int current_byte = 0;
		while (true) {
			current_byte = (s_int_array[m_byte_count]);
			if (current_byte > 255 || current_byte < 0) { break; }
			m_byte_count += 1;
		}
	}
	init_int_array();
}


void Hexr::init_int_vector_bytes() {
	id_ret_type();
	manual_bytes();
	if (m_byte_count == 0) {
		m_byte_count = s_int_vector.size();
	}
	m_vector_used = true;
	delete[] s_int_array;
	s_int_array = &s_int_vector[0];
	init_int_array();
}



// overload int array <<<
// overload hex array >>>

void Hexr::init_hex_array() {
	id_ret_type();
	manual_bytes();
	m_hex_string = "";

	if (m_byte_count == 0) {

		uint64_t nibble_count = 0;
		string first_half = "";
		string second_half = "";
		bool exit_loop = false;
		while (true) {
			m_hex_string += s_hex_array[nibble_count];
			int value1 = 0;
			int value2 = 0;
			bool db = false;
			if (nibble_count > 0) {
				//cout << "m_hex_string length = " << m_hex_string.length() << endl;
				//cout << m_hex_string << endl;
				first_half = m_hex_string.substr(0, nibble_count + 1);
				second_half = m_hex_string.substr(nibble_count + 1, nibble_count * 2);


				auto current_addr = (&((s_hex_array[nibble_count])));
				int value1 = *reinterpret_cast<int *>(current_addr);

				auto next_addr = (&((s_hex_array[nibble_count + 2])));
				int value2 = *reinterpret_cast<int *>(next_addr);

				// old_value > value1 > value2
				if (value2 > (value1 + 9216) || value2 < (value1 - 9216) \
					|| (m_old_value != 0 && (m_old_value >(value1 + 9216) || m_old_value < (value1 - 9216)))) {
					exit_loop = true;
				}
				set_old_value(value1);

			}

			// debug
			// cout << "\nfirst  = " << first_half << "\nsecond = " << second_half << endl; 

			if (first_half.length() != second_half.length()) {

				second_half = first_half[0] + second_half;
				first_half.pop_back();

				if (first_half.length() + 2 < second_half.length()) {
					second_half.erase(first_half.length(), second_half.length());
				}

				m_hex_string = first_half + second_half;
				if (first_half == second_half) {
					nibble_count /= 2;
				}

				set_byte_count(nibble_count);
				//cout << "first\n";
				break;
			}
			else if (first_half.length() > 4 && first_half.compare(2, nibble_count - 2, second_half, 2, nibble_count - 2) == 0) {
				m_hex_string = first_half + second_half;
				nibble_count = (nibble_count + 1) / 2;
				set_byte_count(nibble_count);
				// debug
				//cout << "nibbles = " << nibble_count << "\nfirst  = " << first_half << "\nsecond = " << second_half << endl; 
				break;
			}
			else if ((s_hex_array[nibble_count])[1] == 0) {
				//cout << "second\n";
				set_byte_count(nibble_count);
				m_hex_string = first_half + second_half;
				break;
			}
			nibble_count += 1;
		}
	}
	else {
		m_hex_string = "";
		for (uint64_t i = 0; i < m_byte_count; i++) {
			m_hex_string += s_hex_array[i];
		}
	}

	delete[] m_hex_array;
	m_hex_array = s_hex_array;

	// debug
	//cout << "bytes = " << m_byte_count << endl;for(uint64_t i = 0; i < m_byte_count; i++){cout << m_hex_array[i] << endl;}

	m_start_type = start_hex_array;
	m_converted.out_hs = true;
	m_converted.out_ha = true;

	if (r_ret_type == ha) { hex_string_2_hex_array(); }
	else if (r_ret_type == hs) { hex_array_2_hex_string(); }
	else { hex_array_2_int_array(); }
}



void Hexr::init_hex_vector() {
	id_ret_type();
	m_hex_string = "";
	if (m_man_int > 0) { m_byte_count = m_man_int; }
	else { m_byte_count = s_hex_vector.size(); }
	for (uint64_t i = 0; i < m_byte_count; i++) { m_hex_string += s_hex_vector[i]; }
	m_vector_used = true;
	m_converted.out_hs = true;

	m_start_type = start_hex_string;

	get_path();
	if (r_ret_type == hs) { hex_array_2_hex_string(); }
	else { hex_string_2_hex_array(); }
}


// overload hex array <<<

// Path1: ascii_string > char_array > int_array > hex_array  > hex_string
// Path2: hex_string   > hex_array  > int_array > char_array > ascii_string
int Hexr::get_path() {
	if (int(r_ret_type) > int(m_start_type)) { return 1; } // Path1
	else { return 2; } // Path2
}

void Hexr::id_ret_type() {
	if      (m_ret_type == "as" || m_ret_type == "ascii_string") { r_ret_type = as; }
	else if (m_ret_type == "ca" || m_ret_type == "char_array") { r_ret_type = ca; }
	else if (m_ret_type == "ia" || m_ret_type == "int_array") { r_ret_type = ia; }
	else if (m_ret_type == "ha" || m_ret_type == "hex_array") { r_ret_type = ha; }
	else if (m_ret_type == "hs" || m_ret_type == "hex_string") { r_ret_type = hs; }
	else { m_fast_track = false;   r_ret_type = none; }
}

bool Hexr::done() {
	if (m_converted.out_as == true && m_converted.out_ca == true && m_converted.out_ia == true
		&& (m_converted.out_ha == true && m_converted.out_hs == true)) {
		return true;
	}
	else { return false; }
}

Hexr::~Hexr() {
	if (m_start_type == start_char_array) {
		if (m_vector_in_char_array == false) {
			delete[] m_int_array; delete[] s_int_array;
		}
		delete[] m_hex_array; delete[] s_hex_array;
	}
	else if (m_start_type == start_int_array) {
		delete[] m_char_array; delete[] s_char_array;
		delete[] m_hex_array;  delete[] s_hex_array;
	}
	else if (m_start_type == start_hex_array) {
		delete[] m_char_array; delete[] s_char_array;
		delete[] m_int_array;  delete[] s_int_array;
	}
	else { // vector || string used 
		delete[] m_char_array; delete[] s_char_array;
		delete[] m_int_array;  delete[] s_int_array;
		delete[] m_hex_array;  delete[] s_hex_array;
	}
}

// |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

// Path1: ascii_string > char_array > int_array > hex_array > hex_string

void Hexr::ascii_string_2_char_array() {
	if (m_update_insert != "") { m_insert = m_update_insert; }

	if (m_fast_track == true && r_ret_type != ca) {
		// ascii_string > (skip char_array) > int_array
		delete[] m_int_array; m_int_array = new int[m_byte_count];
		for (int i = 0; i < m_byte_count; i++) {
			m_int_array[i] = int(m_ascii_string[i]);
		}
		set_int_array(m_int_array);
		m_converted.out_ia = true;
		char_array_2_int_array(); // skip next func
	}
	else {
		// default
		if (m_converted.out_ca == false) {
			delete[] m_char_array; m_char_array = new char[m_byte_count];
			for (int i = 0; i < m_byte_count; i++) {
				m_char_array[i] = m_ascii_string[i];
			}
			m_converted.out_ca = true;
		};
		if (((r_ret_type == ca) && m_converted.out_ca == false)
			|| (m_fast_track == true && r_ret_type == ca)) {
			// make this update a string and then allow that string to be called via a ret function											
			for (uint64_t i = 0; i < m_byte_count; i++) {
				m_asm_string = m_asm_string + (m_char_array[i]) + m_insert;
			}; trim_asm_stirng();
		}
		else if (done() == false) {
			char_array_2_int_array();
		}
	}
}

void Hexr::char_array_2_int_array() {
	if (m_update_insert != "") { m_insert = m_update_insert; }

	if (m_fast_track == true && m_converted.out_ia == true
		&& r_ret_type != ia && r_ret_type != ha) {
		// (char_array = skipped) int_array > hex_string
		ostringstream oss;
		m_hex_string = "";
		for (uint64_t i = 0; i < m_byte_count; i++) {
			oss << hex << static_cast<unsigned short>(m_int_array[i]);
		}
		m_hex_string = oss.str();
		m_converted.out_hs = true;
		hex_array_2_hex_string();
	}
	else if (m_fast_track == true && m_converted.out_ca == true && r_ret_type != ia) {
		// char_array > (skip int_array) > hex_array
		delete[] m_hex_array; m_hex_array = new string[m_byte_count];
		for (int i = 0; i < m_byte_count; i++) {
			ostringstream oss;
			oss << hex << static_cast<unsigned short>(m_char_array[i]);
			m_hex_array[i] = oss.str();
		}; m_converted.out_ha = true;
		int_array_2_hex_array();
	}
	else {
		// default
		if (m_converted.out_ia == false) {
			delete[] m_int_array; m_int_array = new int[m_byte_count];
			for (int i = 0; i < m_byte_count; i++) {
				m_int_array[i] = int(m_char_array[i]);
			}
			m_converted.out_ia = true;
			set_int_array(m_int_array);
		};
		if (((r_ret_type == ia))) {
			// make this update a string and then allow that string to be called via a ret function											
			for (uint64_t i = 0; i < m_byte_count; i++) {
				m_asm_string = m_asm_string + to_string(m_int_array[i]) + m_insert;
			}; trim_asm_stirng();

		}
		else if (done() == false) {
			int_array_2_hex_array();
		}
	}
}

// c++ parameters = white
// c++ member functions = custom green (brighter than green, dimer than lime)
// comments = custom dark grey (grey is too bright)
// new/delete = magenta (helps with memory managment)

void Hexr::int_array_2_hex_array() {
	if (m_update_insert != "") { m_insert = m_update_insert; }
	if (m_converted.out_ha == false) {
		delete[] m_hex_array; m_hex_array = new string[m_byte_count];
		for (int i = 0; i < m_byte_count; i++) {
			ostringstream oss;
			oss << hex << static_cast<unsigned short>(m_int_array[i]);
			m_hex_array[i] = oss.str();
		}
	};
	m_converted.out_ha = true;
	if (r_ret_type == ha) {
		for (uint64_t i = 0; i < m_byte_count; i++) { m_asm_string = m_asm_string + m_hex_array[i] + m_insert; }
		trim_asm_stirng();
	}
	else if (done() == false) {
		hex_array_2_hex_string();
	}
}

void Hexr::hex_array_2_hex_string() {
	if (m_update_insert != "") { m_insert = m_update_insert; }

	if (m_fast_track == true && r_ret_type == hs && m_converted.out_hs == true) {
		;// pass, you land here with a "Hex Array Vector input" > "Hex String output"
	}
	else if (m_fast_track == true && r_ret_type != hs && m_converted.out_ha == true) {
		// hex_array > (skip hex_string) > (skip hex_array) > int_array
		hex_array_2_int_array();
	}
	else if (m_converted.out_hs == false) {
		m_hex_string = "";
		for (int i = 0; i < m_byte_count; i++) {
			m_hex_string += m_hex_array[i];
		}
		m_converted.out_hs = true;
	};
	if (r_ret_type == hs) {
		m_asm_string = m_hex_string;
	}
	else if (done() == false && m_converted.out_ha == true) {
		hex_array_2_int_array();
	}
	else if (done() == false) {
		hex_string_2_hex_array();
	}
}


// |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

// Path2: hex_string > hex_array > int_array > char_array > ascii_string

void Hexr::hex_string_2_hex_array() {
	if (m_fast_track == true && r_ret_type != ha) {
		// hex_string > (skip hex_array) > int_array
		if (m_converted.out_ia == false) {
			delete[] m_int_array; int *m_int_array = new int[byte_count()];
			string tmp = "";
			for (int i = 0; i < m_byte_count; i++) {
				tmp = m_hex_string[i * 2]; tmp += m_hex_string[i * 2 + 1];
				m_int_array[i] = stoi(tmp, 0, 16);
			}
			set_int_array(m_int_array);
			m_converted.out_ia = true;
		};

		hex_array_2_int_array();
	}
	else {
		if (m_converted.out_ha == false) {
			delete[] m_hex_array; m_hex_array = new string[m_byte_count];
			string tmp;
			for (int i = 0; i < m_byte_count; i++) {
				// init's string tmp and adds two nibble s and then adds it to an element in the array
				tmp = m_hex_string[i * 2]; tmp += m_hex_string[(i * 2) + 1];	m_hex_array[i] = tmp;
			}
			m_converted.out_ha = true;
		};
		if (r_ret_type == ha) {
			for (uint64_t i = 0; i < m_byte_count; i++) { m_asm_string = m_asm_string + m_hex_array[i] + m_insert; }
			trim_asm_stirng();
		}
		else if (done() == false) {
			hex_array_2_int_array();
		}
	}
}

void Hexr::hex_array_2_int_array() {
	if (m_update_insert != "") { m_insert = m_update_insert; }
	// no m_fast_track to ascii string becaues it is faster to make the char array and then
	// make a pointer point to the array of addresses base address to get the string.

	if (m_converted.out_ia == false) {
		delete[] m_int_array; m_int_array = new int[m_byte_count];
		for (int i = 0; i < m_byte_count; i++) {
			m_int_array[i] = stoi(m_hex_array[i], 0, 16);
		}
		m_converted.out_ia = true;
		set_int_array(m_int_array);
	};
	if (((r_ret_type == ia))) {
		// make this update a string and then allow that string to be called via a ret function											
		for (uint64_t i = 0; i < m_byte_count; i++) {
			m_asm_string = m_asm_string + to_string(m_int_array[i]) + m_insert;
		}; trim_asm_stirng();

	}
	else if (done() == false) {
		int_array_2_char_array();
	}
}


void Hexr::int_array_2_char_array() {
	if (m_update_insert != "") { m_insert = m_update_insert; }
	if (m_converted.out_ca == false) {
		delete[] m_char_array; m_char_array = new char[m_byte_count];
		for (int i = 0; i < m_byte_count; i++) {
			m_char_array[i] = int(m_int_array[i]);
		}
	}; m_converted.out_ca = true;

	char **joined_char = &m_char_array;
	m_ascii_string = *joined_char;
	m_converted.out_as = true;

	if (((r_ret_type == ca))) {
		// make this update a string and then allow that string to be called via a ret function											
		m_asm_string = "";
		for (uint64_t i = 0; i < m_byte_count; i++) {
			m_asm_string = m_asm_string + (m_char_array[i]) + m_insert;
		}; trim_asm_stirng();

	}
	else if (done() == false || r_ret_type == as) {
		char_array_2_ascii_string();
	}
}

void Hexr::char_array_2_ascii_string() {
	if (m_update_insert != "") { m_insert = m_update_insert; }
	// string update was just left for usage consistency even though we don't need it.
	if (m_converted.out_as == false) {
		m_ascii_string = "";
		for (int i = 0; i < m_byte_count; i++) {
			m_ascii_string += m_char_array[i];
		}
		m_converted.out_as = true;
	};
	if (r_ret_type == as) {
		m_asm_string = m_ascii_string;
	}
	else if (done() == false && m_converted.out_ca == true) {
		char_array_2_int_array();
	}
	else if (done() == false) {
		ascii_string_2_char_array();
	}
}

// Path1: ascii_string > char_array > int_array > hex_array > hex_string

// |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

void     Hexr::trim_asm_stirng() { m_asm_string.erase(m_asm_string.end() - m_insert.length(), m_asm_string.end()); }
void     Hexr::set_int_array(int *s_int_array) { m_int_array = s_int_array; }
void     Hexr::set_old_value(int p_value) { m_old_value = p_value; }
void     Hexr::set_byte_count(uint64_t p_byte_cout) { m_byte_count = p_byte_cout; }
void     Hexr::update_insert(string& s_update_insert) { m_update_insert = s_update_insert; }

void     Hexr::bytes(uint64_t s_bytes) { m_byte_count = s_bytes; }
void     Hexr::manual_bytes() {
	if (m_man_int > 0) { m_byte_count = m_man_int; }
	else { m_byte_count = 0; }
}


string   Hexr::ret_type() { return m_ret_type; }
uint64_t Hexr::byte_count() { return m_byte_count; } // in case you need to copy readable code
uint64_t Hexr::bc() { return m_byte_count; } // because bc is shorter
string   Hexr::str() { return s(); } // in case you need to use more readable code
string   Hexr::s() {
	if (r_ret_type != as && r_ret_type != hs) {
		return m_asm_string;
	}
	else if (r_ret_type == hs) {
		return m_asm_string.erase(m_byte_count * 2, m_asm_string.length());
	}
	else { // ascii string
		return m_asm_string.erase(m_byte_count, m_asm_string.length());
	};
}

// --------------------------------------------------------------------------------------------------------	
//  ascii_string > char_array > int_array > hex_array > hex_string

string   Hexr::ascii_string() { return m_ascii_string.erase(m_byte_count, m_ascii_string.length()); }

char    *Hexr::char_array() { return m_char_array; }
cVector  Hexr::char_vector() { cVector retv(m_char_array, m_char_array + m_byte_count); return retv; };

int     *Hexr::int_array() { return m_int_array; }
iVector  Hexr::int_vector() { iVector retv(m_int_array, m_int_array + m_byte_count); return retv; };

string  *Hexr::hex_array() { return m_hex_array; }
sVector  Hexr::hex_vector() { sVector retv(m_hex_array, m_hex_array + m_byte_count); return retv; };

string   Hexr::hex_string() { return m_hex_string.erase(m_byte_count * 2, m_hex_string.length()); }
// --------------------------------------------------------------------------------------------------------	
 
