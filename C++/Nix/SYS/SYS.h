#pragma once

// SYS.h version 1.4.0

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

#include<iostream>
#include<vector>
#include<stdexcept>
#include<unordered_map>
#include<stdio.h>


class SYS
{
private:
	int    m_argc;
	char** m_argv;
	bool m_args_set = false;
	bool m_chain_char_arg = false;

	std::string m_full_path = "";
	std::string m_path = "";
	std::string m_file = "";

	std::string  m_ccaa; // C Char Arg Array

	std::unordered_map<std::string, std::vector<std::string> > m_args; // {base_arg, [sub_args]}

	std::vector<std::string> m_all_args;
	std::string m_all_args_str;

	std::vector<std::string> m_bargs; // base args
	std::string m_bargs_str;
	std::vector< std::vector<std::string> > m_sub_args; // sub args
	std::string m_sub_args_str;

	bool rex_scan(std::string rex, std::vector<std::string> content);
	bool c_arg_chain(char ch);

	// ======================================================================================================================
public:

	SYS(int c_argc, char** c_argv);
	SYS();
	~SYS();
	// -------------------------------------------------------------------------------------------------------------------
	// >>>> args
	SYS set_args(int argc, char** argv, bool chain_char_arg = false);
	void alias(const std::string& s_arg, const char c_arg);
	// -------------------------------------------------------------------------------------------------------------------
	std::vector<std::string> argv();
	int argc();
	std::unordered_map<std::string, std::vector<std::string> > args();
	// -------------------------------------------------------------------------------------------------------------------
	std::string full_path();
	std::string path();
	std::string file();
	// -------------------------------------------------------------------------------------------------------------------
	// Strings
	std::string argv_str();
	std::string all_keys_str();
	std::string all_values_str();
	// -------------------------------------------------------------------------------------------------------------------
	// Return all keys or all values (for each barg)
	std::vector<std::string> all_keys();
	std::vector<std::vector<std::string>> all_values();
	// -------------------------------------------------------------------------------------------------------------------
	// Identify if a barg and/or Value exists
    bool kvp(const std::string& barg);      // does the barg have values
    bool kvp(const char barg);              // does the barg have values
    bool bool_arg(const std::string& barg); // does the barg NOT have values
    bool bool_arg(const char barg); // does the barg NOT have values

	bool has_key(const std::string& barg); // --alias--|
	bool has(const std::string& barg); // -------------|
	bool has(const char barg);

	bool has_arg(const std::string& find_arg);
	bool has_key_value(const std::string& barg, const std::string& value);
	// -------------------------------------------------------------------------------------------------------------------	
	// Return KVP Data
	std::string first(const std::string& barg);  // = sys["barg"][0]
	std::string second(const std::string& barg); // = sys["barg"][1]
	std::string first(const char barg);  // = sys['k'][0]
	std::string second(const char barg); // = sys['k'][1]

	std::vector<std::string> key_values(const std::string& barg); // ---key_values alias--|
	std::vector<std::string> key(const std::string& barg); // ----------------------------|
	std::vector<std::string> key(const char barg);
	std::string key_value(const std::string& barg, int i);
	// -------------------------------------------------------------------------------------------------------------------
	// Almost everything above can be handled using the operator overloading below and is the prefered method
	std::vector<std::string> operator[](const std::string& barg);              // return barg values
	std::vector<std::string> operator[](const char barg);                      // return barg values
	std::string operator[](const int value);                                   // return value by arg location in argv
	bool operator()(const std::string& barg, const std::string& value = "");   // test boolean for barg or KVP
	bool operator()(const char barg, const std::string& value = "");           // test boolean for barg or KVP
	// -------------------------------------------------------------------------------------------------------------------
	bool help();
	// ======================================================================================================================
};

extern SYS sys;

