#pragma once

// SYS.h version 1.5.0

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

// #define DllImport   __declspec( dllimport )
// #define DllExport   __declspec( dllexport )

// ----- Radicalware Libs -------------------
// ----- eXtended STL Functionality ---------
#include "xstring.h"
#include "xvector.h"
#include "xmap.h"
// ----- eXtended STL Functionality ---------
// ----- Radicalware Libs -------------------

class SYS
{
private:
	int    m_argc;
	char** m_argv;
	bool m_kvps_set = false;
	bool m_chain_char_arg = false;

	xstring m_full_path = "";
	xstring m_path = "";
	xstring m_file = "";

	xstring  m_ccaa; // C Char Arg Array

	xmap<xstring, xvector<xstring*>> m_kvps; // {base_arg, [sub_args]}

	xvector<xstring> m_all_args;
	xvector<xstring*> m_keys; // base args
	xvector< xvector<xstring*> > m_sub_args; // sub args

	xvector<xstring> alias_str_arr;
	xstring empty_str = "";

	bool c_arg_chain(char ch);

	// ======================================================================================================================
public:

	SYS(int c_argc, char** c_argv);
	SYS();
	~SYS();
	// -------------------------------------------------------------------------------------------------------------------
	// >>>> args
	SYS set_args(int argc, char** argv, bool chain_char_arg = false);
	void alias(const char c_arg, const xstring& s_arg); // char_arg, string_arg
	// -------------------------------------------------------------------------------------------------------------------
	xvector<xstring> argv(double x = 0, double y = 0, double z = 0);
	int argc();

	xmap<xstring, xvector<xstring*> > kvps();
	xvector<const xstring*> keys();
	xvector<xvector<xstring*>> values();
	// -------------------------------------------------------------------------------------------------------------------
	xstring full_path();
	xstring path();
	xstring file();
	// -------------------------------------------------------------------------------------------------------------------
	bool kvp_arg(const xstring& barg);
	bool kvp_arg(const char barg);
	bool bool_arg(const xstring& barg);
	bool bool_arg(const char barg);
	// -------------------------------------------------------------------------------------------------------------------
	bool has_key(const xstring& barg); // --alias--|
	bool has_key(const char barg); // ------alias--|
	bool has(const xstring& barg); // -------------|
	bool has(const xstring* barg); // -------------|
	bool has(xstring&& barg);      // -------------|
	bool has(const char* barg);    // -------------|
	bool has(const char barg);     // -------------|

	bool has_arg(const xstring& find_arg);
	bool has_key_value(const xstring& barg, const xstring& value);
	bool has_key_value(xstring&& barg, xstring&& value);
	// -------------------------------------------------------------------------------------------------------------------	
	// Return KVP Data
	xstring first(const xstring& barg);  // = sys["barg"][0]
	xstring second(const xstring& barg); // = sys["barg"][1]
	xstring first(const char barg);  // = sys['k'][0]
	xstring second(const char barg); // = sys['k'][1]

	xvector<xstring*> key_values(const xstring& barg); // ---key_values alias--|
	xvector<xstring*> key(const xstring& barg); // ----------------------------|
	xvector<xstring*> key(const char barg);
	xstring key_value(const xstring& barg, int i);
	// -------------------------------------------------------------------------------------------------------------------
	// Almost everything above can be handled using the operator overloading below and is the prefered method
	xvector<xstring*> operator[](const xstring& barg);               // return barg values
	xvector<xstring*> operator[](const char barg);                   // return barg values
	xstring operator[](const int value);                             // return value by arg location in argv
	bool operator()(const xstring& barg, const xstring& value = ""); // test boolean for barg or KVP
	bool operator()(xstring&& barg, xstring&& value = "");           // test boolean for barg or KVP
	bool operator()(const char barg, const xstring& value = "");     // test boolean for barg or KVP
	// -------------------------------------------------------------------------------------------------------------------
	bool help();
	// ======================================================================================================================
};

