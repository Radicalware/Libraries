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
#include<stdexcept>
#include<unordered_map>
#include<stdio.h>
#include<algorithm>
#include<regex>
#include<cstddef>

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

    bool m_args_set = false;

    bool m_kvps_set = false;
    bool m_key_used = false;

    xstring m_full_path;
    xstring m_path;
    xstring m_file;

    xmap<char, int>            m_chr_kvps;   // the int directs to the m_values location
    xstring                    m_chr_lst;
    xmap<xstring*, int>        m_str_kvps;   // the int directs to the m_values location

    xvector<xvector<xstring*>> m_values;     // strings point to m_all_args
    xvector<xstring>           m_all_args;   // where data is extracted (should come from main(int/char**))

    xmap<char, xstring> m_alias;  // User defined and inserted

    void add_value_section(size_t idx_start, size_t idx_end);

    // ======================================================================================================================
public:

    SYS(int c_argc, char** c_argv);
    SYS();
    ~SYS();
    // -------------------------------------------------------------------------------------------------------------------
    // >>>> args
    void set_args(int argc, char** argv);
    void alias(const char c_arg, const xstring& s_arg); // char_arg, string_arg
    // -------------------------------------------------------------------------------------------------------------------
    int argc() const;
    xvector<xstring> argv() const;
    xvector<const xstring*> str_keys() const;
    xstring chr_keys() const;
    // -------------------------------------------------------------------------------------------------------------------
    xstring full_path();
    xstring path();
    xstring file();
    // -------------------------------------------------------------------------------------------------------------------
    xvector<xstring*> key(const xstring& key);
    xvector<xstring*> key(const char key);
    bool key_used() const;
    // -------------------------------------------------------------------------------------------------------------------
    bool has(const xstring& key) const;
    bool has(const xstring* key) const;
    bool has(xstring&& key) const;
    bool has(const char* key) const;
    bool has(const char key) const;
    //// -------------------------------------------------------------------------------------------------------------------
    // Almost everything above can be handled using the operator overloading below and is the prefered method
    xvector<xstring*> operator[](const xstring& key);                     // Return Key-Values
    xvector<xstring*> operator[](const char key);                         // Return Key-Values
    xstring operator[](const int key);                                    // Return value by arg location in argv
    bool operator()(const xstring& key) const;                            // Test boolean for key or Key
    bool operator()(const xstring& key, const xstring& value) const;      // Test boolean for key or KVP
    bool operator()(xstring&& key) const;                                 // Test boolean for key or Key
    bool operator()(xstring&& key, xstring&& value) const;                // Test boolean for key or KVP
    bool operator()(const char key) const;                                // Test boolean for Key
    bool operator()(const char key, const xstring& value) const;          // Test boolean for KVP
    // -------------------------------------------------------------------------------------------------------------------
    bool help();
    // ======================================================================================================================
};

