
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


// -------------------------------------------------------------------------------
// ALERT iterator will majorly slow down your performance if you don't
// optimize your compiler settings "-O2", else it will increase speed when
// not on windows (windows will give you even speeds with optimization else lag)
// also, on windows, be sure to remove the debugging for the iterator. 
// -------------------------------------------------------------------------------



#include "../include/SYS.h"

#include<iostream>
#include<vector>
#include<stdexcept>
#include<unordered_map>
#include<stdio.h>
#include<algorithm>
#include<regex>
#include<cstddef>

SYS::SYS(int c_argc, char** c_argv) :
	m_argc(c_argc), m_argv(c_argv) {
	(this)->set_args(m_argc, m_argv);
};


SYS::SYS() {};

SYS::~SYS() {};

bool SYS::c_arg_chain(char ch) {
	if (m_ccaa.size())
		return (std::find(m_ccaa.begin(), m_ccaa.end(), ch) != m_ccaa.end());
	return false;
}

// ======================================================================================================================
// >>>> args

SYS SYS::set_args(int argc, char** argv, bool chain_char_arg) {

	m_full_path = xstring(argv[0]);
	size_t program_split_loc = m_full_path.find_last_of("/\\");
	m_path = m_full_path.substr(0, program_split_loc + 1);
	m_file = m_full_path.substr(program_split_loc + 1, m_full_path.size() - 1);

	m_argc = argc;
	m_chain_char_arg = chain_char_arg;
	if (argc == 1 || m_kvps_set) { return *this; }

	bool key_start = (argv[1][0] == '-');

	bool sub_arg = false;
	bool first_header = true;
	int first_sub = 0;
	xvector<xstring*> prep_sub_args; // STRs come from m_all_args

	xstring* current_base = &empty_str; // argv > m_all_args >> current_base

	m_all_args.insert(m_all_args.begin(), &argv[0], &argv[argc]);

	auto add_chain_char_arg = [&]()->void {
		if ((*current_base)[0] == '-' && (*current_base)[1] != '-') {

			xstring barg { (*current_base)[0], (*current_base)[current_base->size() - 1] };
			m_kvps.insert(std::make_pair(barg, prep_sub_args));
		}
	};

	for (int arg_idx = 0; arg_idx <= argc; arg_idx++) { // loop through argv

		if (arg_idx == argc) { // executes at the end of the args
			if (prep_sub_args.size() == 0) { prep_sub_args.push_back(nullptr); }
			// last call to append prep_sub_args to KVP
			if (m_kvps.has(*current_base)) {
				for (xstring* str : prep_sub_args)
					m_kvps.at(*current_base).push_back(str);
			}
			else {
				if (current_base->size()) {
					if (chain_char_arg && (*current_base)[1] != '-') {
						add_chain_char_arg();
					}
					else {
						m_kvps.insert(std::make_pair(*current_base, prep_sub_args));
					}
				}
				else {
					m_kvps.insert(std::make_pair("none", prep_sub_args)); // note: none isn't -none
				}
			}
		}
		else if (arg_idx < argc) { // executes at a new barg
			if (argv[arg_idx][0] == '-') {
				if (prep_sub_args.size() > 0) { // append sub args to the last barg
					m_sub_args.push_back(prep_sub_args);
					if (m_kvps.has(*current_base)) { // if the barg was already specified
						m_kvps.insert(std::make_pair(*current_base, m_sub_args.back()));
					}
					else {
						if (chain_char_arg && current_base->size() && (*current_base)[1] != '-') 
							add_chain_char_arg();
						else 
							m_kvps.insert(std::make_pair(*current_base, m_sub_args.back()));
					}
					prep_sub_args.clear();
				}
				else if (!chain_char_arg) { // barg does not have any sub-args
					m_kvps.insert(std::make_pair(m_all_args[static_cast<size_t>(arg_idx) - 1], xvector<xstring*>({ &empty_str })));
				}
				current_base = &empty_str;
				current_base = &m_all_args[arg_idx];

				m_keys.push_back(&m_all_args[arg_idx]);
			}
			else { // appends new sub args
				prep_sub_args.push_back(&m_all_args[arg_idx]);
			}
		}
	}

	// C Char Arg Array only occurs when kvp() == false
	if (m_chain_char_arg) {
		for (xstring* key : m_keys) {
			if (key->size() > 1 && ((*key)[0] == '-' && (*key)[1] != '-')) {
				m_ccaa += key->substr(1, key->size() - 1); // grab all keys for a joined set; ie: -abc
			}
		}
		for (char char_val : m_ccaa) {
			alias_str_arr.push_back(xstring({ '-',char_val }));
			m_kvps.insert(std::make_pair(alias_str_arr.back(), xvector<xstring*>({&empty_str})));
		}
	}

	m_kvps.relocate();
	m_kvps_set = true;
	
	return *this;
}


void SYS::alias(const char c_arg, const xstring& s_arg) {
	// char_arg, string_arg
	// add the alias for the char or string version of the arg

	if (m_kvps.has(s_arg)) {
		m_kvps.insert(std::make_pair(xstring({ '-',c_arg }), m_kvps.at(s_arg)));
		alias_str_arr << xstring({ '-',c_arg });
		m_keys.push_back(&alias_str_arr[alias_str_arr.size() - 1]);
		m_ccaa += c_arg;

	}
	else if (this->has(c_arg)) {
		alias_str_arr << s_arg;
		m_keys.push_back(&alias_str_arr[alias_str_arr.size() - 1]);
		m_kvps.insert(std::make_pair(s_arg, this->key(c_arg)));
	}
}


// -------------------------------------------------------------------------------------------------------------------
xvector<xstring> SYS::argv(double x, double y, double z) { 
	if (x == 0 && y == 0 && z == 0)
		return m_all_args;
	else
		return m_all_args(x, y, z);
}
int SYS::argc() { return m_argc; }
xmap<xstring, xvector<xstring*> > SYS::kvps() { return m_kvps; }
xvector<const xstring*> SYS::keys() { return m_kvps.keyStore(); }
xvector<xvector<xstring*>> SYS::values() { return m_sub_args; }
// -------------------------------------------------------------------------------------------------------------------
xstring SYS::full_path() { return m_full_path; }
xstring SYS::path() { return m_path; }
xstring SYS::file() { return m_file; }
// -------------------------------------------------------------------------------------------------------------------
bool SYS::kvp_arg(const xstring& barg) {
	if (m_kvps.key(barg).size())
		return true;
	return false;
}

bool SYS::kvp_arg(const char c_barg) {
	if (std::find(m_ccaa.begin(), m_ccaa.end(), c_barg) != m_ccaa.end())
		return false;
	return true;
}

bool SYS::bool_arg(const xstring& barg) {
	if (m_kvps.key(barg).size())
		return false;
	return true;
}

bool SYS::bool_arg(const char c_barg) {
	if (std::find(m_ccaa.begin(), m_ccaa.end(), c_barg) != m_ccaa.end())
		return true;
	return false;
}
// -------------------------------------------------------------------------------------------------------------------

bool SYS::has(const xstring& barg) {
	if (std::find(m_keys.begin(), m_keys.end(), &barg) != m_keys.end())
		return true;
	return this->c_arg_chain(barg[1]);
};

bool SYS::has(const xstring* barg) {
	for (typename xvector<xstring*>::iterator it = m_keys.begin(); it != m_keys.end(); it++) {
		if (**it == *barg)
			return true;
	}
	return this->c_arg_chain((*barg)[1]);
}
bool SYS::has(xstring&& barg)
{
	for (typename xvector<xstring*>::iterator it = m_keys.begin(); it != m_keys.end(); it++) {
		if (**it == barg)
			return true;
	}
	return this->c_arg_chain(barg[1]);
}

bool SYS::has(const char* barg)
{
	return this->has(xstring(barg));
}


bool SYS::has(const char barg) {
	return (std::find(m_ccaa.begin(), m_ccaa.end(), barg) != m_ccaa.end());
};

bool SYS::has_key(const xstring& barg) { return m_kvps.has(barg); }
bool SYS::has_key(const char barg) { return m_ccaa.has(barg); }

bool SYS::has_arg(const xstring& find_arg) {
	if (std::find(m_all_args.begin(), m_all_args.end(), find_arg) != m_all_args.end()) 
		return true;
	return false;
}

bool SYS::has_key_value(const xstring& barg, const xstring& value) {
	if (m_kvps.at(barg).has(value))
		return true;
	return false;
}
bool SYS::has_key_value(xstring&& barg, xstring&& value)
{
	return this->has_key_value(barg, value);;
}
// -------------------------------------------------------------------------------------------------------------------
xstring SYS::first(const xstring& barg) {
	if (m_kvps.has(barg) && m_kvps.at(barg).size())
		return *m_kvps.at(barg)[0];
	return xstring();
}

xstring SYS::second(const xstring& barg) {
	if (m_kvps.has(barg) && m_kvps.at(barg).size() >= 2)
		return *m_kvps.at(barg)[1];
	return xstring();
};

xstring SYS::first(const char i_key) {
	return this->first(xstring({ '-', i_key }));
}
xstring SYS::second(const char i_key) {
	return this->second(xstring({ '-', i_key }));
};

xvector<xstring*> SYS::key_values(const xstring& barg) { return this->key(barg); } // --alias of barg--|
xvector<xstring*> SYS::key(const xstring& barg) { return m_kvps.at(barg); }  //  ----------------------|
xvector<xstring*> SYS::key(const char barg) { return m_kvps.at(xstring{ '-',barg }); }
xstring SYS::key_value(const xstring& barg, int i) { return *m_kvps.at(barg)[i]; }

// -------------------------------------------------------------------------------------------------------------------

xvector<xstring*> SYS::operator[](const xstring& barg) { return m_kvps.at(barg); }
xvector<xstring*> SYS::operator[](const char barg) { return m_kvps.at(xstring({ '-',barg })); }

xstring SYS::operator[](int value) {
	if (m_all_args.size() <= value) {
		return xstring("");
	}
	else {
		return m_all_args[value];
	}
}

bool SYS::operator()(const xstring& barg, const xstring& value) {
	if (m_keys.has(barg)) {
		if (value.size()) {
			if (m_kvps.at(barg).has(value)) {
				return true;
			}
		}
		else {
			return true;
		}
	}

	if (barg.size() == 2)
		return this->c_arg_chain(barg[1]);
	return false;
}

bool SYS::operator()(xstring&& barg, xstring&& value)
{
	return (*this)(barg, value);
}

bool SYS::operator()(const char c_barg, const xstring& value) {
	if (std::find(m_ccaa.begin(), m_ccaa.end(), c_barg) != m_ccaa.end()) {
		if (value.size()) {
			if (m_kvps.at(xstring({ '-', c_barg })).has(value)) {
				return true;
			}
		}
		else {
			return true;
		}
	}
	return false;
}

// -------------------------------------------------------------------------------------------------------------------
bool SYS::help() {
	return (m_keys.match_one(R"(^[-]{1,2}[hH]((elp)?)$)"));
}
// ======================================================================================================================

