
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

#include "sys.h"
#include "re.h" // From github.com/Radicalware
                // re.h has no non-std required libs
                // This is the only non-std lib required for SYS.h


#include<iostream>
#include<vector>
#include<stdexcept>

#include<unordered_map>

#include<stdio.h>

SYS sys;

SYS::SYS(int c_argc, char** c_argv) :
    m_argc(c_argc), m_argv(c_argv) {
    (this)->set_args(m_argc, m_argv);
};


SYS::SYS() {};

SYS::~SYS() {};

// ============================================================================================
// >>>> args

SYS SYS::set_args(int argc, char** argv) {
    

    bool key_start = (argv[1][0] == '-' || argv[1][0] == '/');

    m_all_args.resize(argc);
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
                if (!this->has(m_all_args[arg])) { // if the key doesn't already exist
                    m_args.insert(std::make_pair(m_all_args[arg], std::vector<std::string>{""}));
                }
                if (prep_sub_arg.size() > 0) { // append sub args to the last key
                    m_sub_args.push_back(prep_sub_arg);
                    if (m_args.count(current_base)) { // if the key was alreayd specified
                        for (std::string str : prep_sub_arg)
                            m_args.at(current_base).push_back(str);
                    }
                    else {
                        m_args.insert(std::make_pair(current_base, prep_sub_arg));
                    }
                    for (std::string str : prep_sub_arg)m_sub_args_str += str + ' ';
                    prep_sub_arg.clear();
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

    return *this;
}

// All arg usage which which should only be used for small C++ tools.

std::vector<std::string> SYS::argv() { return m_all_args; }
int SYS::argc() { return m_all_args.size(); }

std::string SYS::operator[](int value) {
    if (m_all_args.size() <= value) {
        return std::string("");
    }
    else {
        return m_all_args[value];
    }
}
std::unordered_map<std::string, std::vector<std::string> > SYS::args() { return m_args; }

// -------------------------------------------------------------------------------

std::string SYS::argv_str() { return m_all_args_str; }

bool SYS::has_arg(const std::string& find_arg) {
    for (auto&arg : m_all_args) {
        if (arg == find_arg)
            return true;
    }return false;
}


std::vector<std::string> SYS::keys() { return m_bargs; }
std::string SYS::keys_str() { return m_bargs_str; }

std::vector< std::vector<std::string> > SYS::key_values() { return m_sub_args; }
std::string SYS::key_values_str() { return m_sub_args_str; }

// -------------------------------------------------------------------------------
// These are the arg functions you will mostly use
// 1st you will identify if the key exist with "has()"
// 2nd you will either return it's values with "key_values()"
//     or you will get the bool for the existence of its value
//     "has_key_value()" for control flow of the program

std::string SYS::key_value(const std::string& key, int i) { return m_args.at(key)[i]; }

std::vector<std::string> SYS::key(const std::string& key) { return m_args.at(key); }
std::vector<std::string> SYS::key_values(const std::string& key) { return this->key(key); }

std::vector<std::string> SYS::operator[](const std::string& key) { return m_args.at(key); }

bool SYS::has(const std::string& key) {
    std::vector<std::string>::iterator iter;
    for (iter = m_bargs.begin(); iter != m_bargs.end(); ++iter) {
        if (*iter == key) {
            return true;
        }
    }return false;
};
bool SYS::key_value(const std::string& key, const std::string& value) {
    for (std::vector<std::string>::const_iterator iter = m_args.at(key).begin(); \
        iter != m_args.at(key).end(); iter++) {
        if (*iter == value) {
            return true;
        }
    }return false;
}
bool SYS::has_key_value(const std::string& key, const std::string& value) {
    return this->key_value(key, value);
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

// -------------------------------------------------------------------------------
template<class T>
void SYS::p(T input) { std::cout << std::endl << "------\n" << input << std::endl; }
void SYS::d(int input) { std::cout << std::endl << "---{dbg: " << input << "}---" << std::endl; }
// -------------------------------------------------------------------------------

