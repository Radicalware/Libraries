
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

#include "SYS.h"

#include<iostream>
#include<vector>
#include<stdexcept>
#include<unordered_map>
#include<stdio.h>
#include<algorithm>
#include<regex>
#include <cstddef>

SYS sys;

SYS::SYS(int c_argc, char** c_argv) :
	m_argc(c_argc), m_argv(c_argv) {
	(this)->set_args(m_argc, m_argv);
};


SYS::SYS() {};

SYS::~SYS() {};


bool SYS::rex_scan(std::string rex, std::vector<std::string> content) {

	std::regex pattern(rex);
	for (std::vector<std::string>::const_iterator iter = content.begin(); iter != content.end(); iter++) {
		if (std::regex_search(*iter, pattern)) {
			return true;
		}
	}
	return false;
}


bool SYS::c_arg_chain(char ch) {
	if (m_ccaa.size()) {
		return (std::find(m_ccaa.begin(), m_ccaa.end(), ch) != m_ccaa.end());
	} else {
		return false;
	}
}

// ======================================================================================================================
// >>>> args

SYS SYS::set_args(int argc, char** argv, bool chain_char_arg) {
	
	m_full_path = std::string(argv[0]);
	size_t program_split_loc = m_full_path.find_last_of("/\\");
	m_path = m_full_path.substr(0, program_split_loc+1);
	m_file = m_full_path.substr(program_split_loc+1,m_full_path.size()-1);

	m_argc = argc;
	m_chain_char_arg = chain_char_arg;
	if (argc == 1 || m_args_set) { return *this; }

	bool key_start = (argv[1][0] == '-' || argv[1][0] == '/');

	bool sub_arg = false;
	bool first_header = true;
	int first_sub = 0;
	std::vector<std::string> prep_sub_arg;
	std::string current_base;

	m_all_args.insert(m_all_args.begin(), &argv[0], &argv[argc]);

	auto add_chain_char_arg = [&]()->void {
		if ((current_base[0] == '-' || current_base[0] == '/') &&
			(current_base[1] != '-' && current_base[1] != '/')) {

			std::string barg = { current_base[0], current_base[current_base.size() - 1] };
			m_args.insert(std::make_pair(barg, prep_sub_arg));
		}
	};

	for (int arg = 1; arg <= argc; arg++) { // loop through argv

		if (arg == argc) { // executes at the end of the args
			if (prep_sub_arg.size() == 0) { prep_sub_arg.push_back(""); }
			// last call to append prep_sub_arg to KVP
			if (m_args.count(current_base)) {
				for (std::string str : prep_sub_arg)
					m_args.at(current_base).push_back(str);
			} else {
				if (chain_char_arg && current_base[1] != '-' && current_base[1] != '/') {
					add_chain_char_arg();
				} else {
					m_args.insert(std::make_pair(current_base, prep_sub_arg));
				}
			}
			for (std::string str : prep_sub_arg)m_sub_args_str += str + ' ';
		} else if (arg < argc) { // executes at a new barg
			if (argv[arg][0] == '-' || argv[arg][0] == '/') {
				if (prep_sub_arg.size() > 0) { // append sub args to the last barg
					m_sub_args.push_back(prep_sub_arg);
					if (m_args.count(current_base)) { // if the barg was already specified
						for (std::string str : prep_sub_arg)
							m_args.at(current_base).push_back(str);
					} else {
						if (chain_char_arg && current_base[1] != '-' && current_base[1] != '/') {
							add_chain_char_arg();
						} else {
							m_args.insert(std::make_pair(current_base, prep_sub_arg));
						}
					}
					for (std::string str : prep_sub_arg)m_sub_args_str += str + ' ';
					prep_sub_arg.clear();
				} else { // barg does not have any sub-args
					m_args.insert(std::make_pair(m_all_args[arg-1], std::vector<std::string>{""}));
				}
				current_base = m_all_args[arg];
				m_bargs.push_back(current_base);
			} else { // appends new sub args
				prep_sub_arg.push_back(m_all_args[arg]);
			}
		}
	}

	for (std::string& str : m_all_args) m_all_args_str += str + " ";
	for (std::string& str : m_bargs) m_bargs_str += str + ' ';
	
	// C Char Arg Array only occurs when kvp() == false
	if (m_chain_char_arg) {
		for (std::string& barg : m_bargs) {
			if (barg.size() > 1 && (barg[0] == '-' || barg[0] == '/') \
				&& (barg[1] != '-' && barg[1] != '/')) {
				m_ccaa += barg.substr(1, barg.size() - 1);
			}
		}
	}

	m_args_set = true;

	return *this;
}

// -------------------------------------------------------------------------------------------------------------------
std::vector<std::string> SYS::argv() { return m_all_args; }
int SYS::argc() { return m_argc; }
std::unordered_map<std::string, std::vector<std::string> > SYS::args() { return m_args; }
// -------------------------------------------------------------------------------------------------------------------
std::string SYS::full_path() { return m_full_path; }
std::string SYS::path() { return m_path; }
std::string SYS::file() { return m_file; }
// -------------------------------------------------------------------------------------------------------------------
std::string SYS::argv_str() { return m_all_args_str; }
std::string SYS::keys_str() { return m_bargs_str; }
std::string SYS::key_values_str() { return m_sub_args_str; }
// -------------------------------------------------------------------------------------------------------------------
std::vector<std::string> SYS::keys() { return m_bargs; }
std::vector< std::vector<std::string> > SYS::key_values() { return m_sub_args; }
// -------------------------------------------------------------------------------------------------------------------


bool SYS::kvp(const std::string& barg) {
	if (this->has(barg)) {
		return m_args.at(barg)[0].size();
	}return false;
}

bool SYS::bool_arg(const std::string& barg) {
	if (this->has(barg)) {
		return (!m_args.at(barg)[0].size());
	}return true;
}

bool SYS::has(const std::string& barg) {
	if (std::find(m_bargs.begin(), m_bargs.end(), barg) != m_bargs.end()) {
		return true;
	}return this->c_arg_chain(barg[1]);
};

bool SYS::has_key(const std::string& barg) { return this->has(barg); }

bool SYS::has_arg(const std::string& find_arg) {
	if (std::find(m_all_args.begin(), m_all_args.end(), find_arg) != m_all_args.end()) {
		return true;
	}return false;
}

bool SYS::has_key_value(const std::string& barg, const std::string& value) {
	if (std::find(m_args.at(barg).begin(), m_args.at(barg).end(), value) != m_args.at(barg).end()) {
		return true;
	}return false;
}
// -------------------------------------------------------------------------------------------------------------------
std::string SYS::first(const std::string& barg) {
	if (this->kvp(barg)) {
		return m_args.at(barg)[0];
	} return "";
}

std::string SYS::second(const std::string& barg) {
	if (this->kvp(barg)) {
		return m_args.at(barg)[1];
	} return "";
};

std::string SYS::first(const char i_key) {
	return this->first(std::string({ '-',i_key }));
}
std::string SYS::second(const char i_key) {
	return this->second(std::string({ '-',i_key }));
};

std::vector<std::string> SYS::key_values(const std::string& barg) { return this->key(barg); } // --alias of barg--|
std::vector<std::string> SYS::key(const std::string& barg) { return m_args.at(barg); } //  ----------------------|
std::string SYS::key_value(const std::string& barg, int i) { return m_args.at(barg)[i]; }

// -------------------------------------------------------------------------------------------------------------------

std::vector<std::string> SYS::operator[](const std::string& barg) { return m_args.at(barg); }
std::vector<std::string> SYS::operator[](const char barg) { return m_args.at(std::string({ '-',barg})); }

std::string SYS::operator[](int value) {
	if (m_all_args.size() <= value) {
		return std::string("");
	} else {
		return m_all_args[value];
	}
}

bool SYS::operator()(const std::string& barg, const std::string& value) {


	if (this->rex_scan(barg, m_bargs)) {
		if (value.size()) {
			if (this->rex_scan(value, m_args.at(barg))) {
				return true;
			}
		} else {
			return true;
		}
	}

	if (barg.size() == 2) {
		return this->c_arg_chain(barg[1]);
	}return false;
}

bool SYS::operator()(const char c_barg, const std::string& value) {
	if (std::find(m_ccaa.begin(), m_ccaa.end(), c_barg) != m_ccaa.end()){
		if (value.size()) {
			if (this->rex_scan(value, m_args.at(std::string({ '-', c_barg })))) {
				return true;
			}
		} else {
			return true;
		}
	}return false;
}

// -------------------------------------------------------------------------------------------------------------------
bool SYS::help() {
	return (this->rex_scan(R"(^[-]{1,2}[hH]((elp)?)$)", m_bargs));
}
// ======================================================================================================================

