#pragma once

// SYS.h version 1.1.0

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

#include "re.h" // From github.com/Radicalware
				// re.h has no non-std required libs
				// This is the only non-std lib required for SYS.h


#include<iostream>
#include<vector>
#include<stdexcept>
#include<unordered_map>
#include <stdio.h>


class SYS
{
private:
	std::string m_command;
	std::string m_file_name;
	std::string m_file_data = "  ";

	char m_last_read = 'n';
	char m_write_method = 'a';

	int    m_argc;
	char** m_argv;
	bool m_args_set = false;

	std::unordered_map<std::string, std::vector<std::string> > m_args; // {base_arg, [sub_args]}

	std::vector<std::string> m_all_args;
	std::string m_all_args_str;

	std::vector<std::string> m_bargs; // base args
	std::string m_bargs_str;
	std::vector< std::vector<std::string> > m_sub_args; // sub args
	std::string m_sub_args_str;


	// ======================================================================================================================
public:

	SYS(int c_argc, char** c_argv);
	SYS();
	~SYS();

	// -------------------------------------------------------------------------------------------------------------------
	// >>>> args
	SYS set_args(int argc, char** argv);
	// -------------------------------------------------------------------------------------------------------------------
	std::vector<std::string> argv();
	int argc();
	std::unordered_map<std::string, std::vector<std::string> > args();
	// -------------------------------------------------------------------------------------------------------------------
	std::string argv_str();
	std::string keys_str();
	std::string key_values_str();
	// -------------------------------------------------------------------------------------------------------------------
	// no input means return all keys or all key_values;
	std::vector<std::string> keys();
	std::vector<std::vector<std::string>> key_values();
	// -------------------------------------------------------------------------------------------------------------------
	bool has(const std::string& key);
	bool has_key(const std::string& key);
	bool has_arg(const std::string& find_arg);
	bool has_key_value(const std::string& key, const std::string& value);
	// -------------------------------------------------------------------------------------------------------------------
	std::string key_value(const std::string& key, int i); 
	std::vector<std::string> key_values(const std::string& key); // -----------|
	std::vector<std::string> key(const std::string& key); // key_values alias--|
	// -------------------------------------------------------------------------------------------------------------------

	std::vector<std::string> operator[](const std::string& key);              // return key values
	std::string operator[](const int value);                                  // return value by arg location in argv
	bool operator()(const std::string& key, const std::string& value = "");   // test boolean for key or KVP

	// -------------------------------------------------------------------------------------------------------------------
	template<class T = std::string>
	void p(T str);
	void d(int i = 0);
	// ======================================================================================================================
};

extern SYS sys;
