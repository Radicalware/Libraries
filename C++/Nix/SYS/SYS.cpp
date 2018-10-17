
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
#include "re.h" // From github.com/Radicalware
				// re.h has no non-std required libs
				// This is the only non-std lib required for SYS.h


#include<iostream>
#include<vector>
#include<stdexcept>
#include<unordered_map>
#include<stdio.h>
#include<algorithm>

SYS sys;

SYS::SYS(int c_argc, char** c_argv) :
	m_argc(c_argc), m_argv(c_argv) {
	(this)->set_args(m_argc, m_argv);
};


SYS::SYS() {};

SYS::~SYS() {};

// ======================================================================================================================
// >>>> args

SYS SYS::set_args(int argc, char** argv) {
	if (argc == 1 || m_args_set) { return *this; }

	bool key_start = (argv[1][0] == '-' || argv[1][0] == '/');

	bool sub_arg = false;
	bool first_header = false;
	int first_sub = 0;
	std::vector<std::string> prep_sub_arg;
	std::string current_base;

	m_all_args.insert(m_all_args.begin(), &argv[0], &argv[argc]);

	for (int arg = 0; arg <= argc; arg++) { // loop through argv

		if (arg == argc) { // executes at the end of the args
			if (prep_sub_arg.size() == 0) { prep_sub_arg.push_back(std::string("")); }
			// last call to append prep_sub_arg to KVP
			if (m_args.count(current_base)) {
				for (std::string str : prep_sub_arg)
					m_args.at(current_base).push_back(str);
			}
			else {
				m_args.insert(std::make_pair(current_base, prep_sub_arg));
			}
			for (std::string str : prep_sub_arg)m_sub_args_str += str + ' ';
		}
		else if (arg < argc) { // executes at a new key
			if (argv[arg][0] == '-' || argv[arg][0] == '/') {
				if (prep_sub_arg.size() > 0) { // append sub args to the last key
					m_sub_args.push_back(prep_sub_arg);
					if (m_args.count(current_base)) { // if the key was already specified
						for (std::string str : prep_sub_arg)
							m_args.at(current_base).push_back(str);
					}
					else {
						m_args.insert(std::make_pair(current_base, prep_sub_arg));
					}
					for (std::string str : prep_sub_arg)m_sub_args_str += str + ' ';
					prep_sub_arg.clear();
				}
				else { // key does not have any sub-args
					m_args.insert(std::make_pair(m_all_args[arg], std::vector<std::string>{""}));
				}
				current_base = m_all_args[arg];
				m_bargs.push_back(current_base);
			}
			else { // appends new sub args
				prep_sub_arg.push_back(m_all_args[arg]);
			}
		}
	}

	for (std::string& str : m_all_args) m_all_args_str += str + " ";
	for (std::string& str : m_bargs) m_bargs_str += str + ' ';

	m_args_set = true;
	return *this;
}
// -------------------------------------------------------------------------------------------------------------------

std::vector<std::string> SYS::argv() { return m_all_args; }
int SYS::argc() { return m_argc; }
std::unordered_map<std::string, std::vector<std::string> > SYS::args() { return m_args; }
// -------------------------------------------------------------------------------------------------------------------
std::string SYS::argv_str() { return m_all_args_str; }
std::string SYS::keys_str() { return m_bargs_str; }
std::string SYS::key_values_str() { return m_sub_args_str; }
// -------------------------------------------------------------------------------------------------------------------
std::vector<std::string> SYS::keys() { return m_bargs; }
std::vector< std::vector<std::string> > SYS::key_values() { return m_sub_args; }
// -------------------------------------------------------------------------------------------------------------------
bool SYS::has(const std::string& key) {
	if (std::find(m_bargs.begin(), m_bargs.end(), key) != m_bargs.end()){
		return true;
	}return false;
};

bool SYS::has_key(const std::string& key) { return this->has(key); }

bool SYS::has_arg(const std::string& find_arg) {
	if (std::find(m_all_args.begin(), m_all_args.end(), find_arg) != m_all_args.end()) {
		return true;
	}return false;
}

bool SYS::has_key_value(const std::string& key, const std::string& value) {
	if (std::find(m_args.at(key).begin(), m_args.at(key).end(), value) != m_args.at(key).end()) {
		return true;
	}return false;
}
// -------------------------------------------------------------------------------------------------------------------
std::string SYS::key_value(const std::string& key, int i) { return m_args.at(key)[i]; }
std::vector<std::string> SYS::key_values(const std::string& key) { return this->key(key); } // ----------------|
std::vector<std::string> SYS::key(const std::string& key) { return m_args.at(key); } // alias of key_values ---|
// -------------------------------------------------------------------------------------------------------------------


std::vector<std::string> SYS::operator[](const std::string& key) { return m_args.at(key); }

std::string SYS::operator[](int value) {
	if (m_all_args.size() <= value) {
		return std::string("");
	}
	else {
		return m_all_args[value];
	}
}

bool SYS::operator()(const std::string& key, const std::string& value) {
	if (std::find(m_bargs.begin(), m_bargs.end(), key) != m_bargs.end()) {
		if (value.size()) {
			if (std::find(m_args.at(key).begin(), m_args.at(key).end(), value) != m_args.at(key).end()) {
				return true;
			}
			else {
				return false;
			}
		}
		else {
			return true;
		}
	}
	else {
		return false;
	}
}

// -------------------------------------------------------------------------------------------------------------------
template<class T>
void SYS::p(T input) { std::cout << std::endl << "------\n" << input << std::endl; }
void SYS::d(int input) { std::cout << std::endl << "---{dbg: " << input << "}---" << std::endl; }
// ======================================================================================================================

