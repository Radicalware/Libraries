#pragma once
#pragma warning (disable : 26444) // allow anynomous objects

/*
* Copyright[2019][Joel Leagues aka Scourge]
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
#include<string>
#include<vector>
#include<utility>
#include<regex>
#include<functional> 
#include<algorithm>
#include<locale>
#include<memory>
#include<initializer_list>
#include<iostream>
#include<string.h>

#include "xvector.h"

namespace rxm {
    using namespace std::regex_constants;
    using type = syntax_option_type;
}
class xstring;
template<typename T>
xstring to_xstring(const T& num);

class xstring : public std::string
{
public:
    using ull = unsigned long long;

    xstring();

    xstring(std::string&& str) noexcept;
    xstring(const std::string& str);

    xstring(xstring&& str) noexcept;
    xstring(const xstring& str);

    xstring(const char* str);
    xstring(const char i_char, const int i_int = 1);
    xstring(const int i_int, const char i_char);
    xstring(std::initializer_list<char> lst);

    void operator=(const xstring& other);
    void operator=(xstring&& other) noexcept;
    void operator=(const char* other);

    //void operator+=(const size_t& val);
    xstring operator+(const char* str) const;
    void operator+=(const char* str);
    void operator+=(const xstring& str);

    bool operator==(const char* other) const;

    void print() const;
    void print(int num) const;
    void print(const xstring& front, const xstring& end = "") const;

    std::string to_string() const;

    xstring upper() const;
    xstring lower() const;

    xstring operator*(int total);
    xstring operator*=(int total);

    // =================================================================================================================================
    
    xvector<xstring> split(size_t loc) const;
    xvector<xstring> split(const xstring& in_pattern, rxm::type mod = rxm::ECMAScript) const;
    xvector<xstring> split(xstring&& in_pattern, rxm::type mod = rxm::ECMAScript) const;
    xvector<xstring> split(const char splitter, rxm::type mod = rxm::ECMAScript) const;

    //// =================================================================================================================================
    //   match is based on regex_match
    bool match(const xstring& in_pattern, rxm::type mod = rxm::ECMAScript) const;
    bool match_line(const xstring& in_pattern, rxm::type mod = rxm::ECMAScript) const;
    bool match_lines(const xstring& in_pattern, rxm::type mod = rxm::ECMAScript) const;

    //   scan is based on regex_search
    bool scan(const char in_pattern, rxm::type mod = rxm::ECMAScript) const;
    bool scan(const xstring& in_pattern, rxm::type mod = rxm::ECMAScript) const;
    bool scan_line(const xstring& in_pattern, rxm::type mod = rxm::ECMAScript) const;
    bool scan_lines(const xstring& in_pattern, rxm::type mod = rxm::ECMAScript) const;

    // exact match (no regex)
    bool is(const xstring& other) const;
    size_t hash() const;

    // =================================================================================================================================

    bool has_non_ascii() const;
    xstring remove_non_ascii() const;

    // =================================================================================================================================
private:
    // re::search & re::findall use grouper/iterator, don't use them via the namespace directly
    xvector<xstring> grouper(const xstring& content, xvector<xstring>& ret_vector, const std::regex& pattern) const;
    xvector<xstring> iterate(const xstring& content, xvector<xstring>& ret_vector, const std::regex& pattern) const;
    // --------------------------------------------------------------------------------------------------------------------------------
public:
    std::vector<xstring> findall(const std::string& in_pattern, rxm::type mod = rxm::ECMAScript, const bool group = false) const;
    // =================================================================================================================================

    bool has(const char var_char, rxm::type mod = rxm::ECMAScript) const;
    bool lacks(const char var_char, rxm::type mod = rxm::ECMAScript) const;
    unsigned long long count(const char var_char, rxm::type mod = rxm::ECMAScript) const;
    unsigned long long count(const xstring& in_pattern, rxm::type mod = rxm::ECMAScript) const;

    // =================================================================================================================================

    xstring sub(const std::string& in_pattern, const std::string& replacement, rxm::type mod = rxm::ECMAScript) const;
    xstring strip(); // this updates *this as well as return *this

    // =================================================================================================================================

    xstring operator()(double x = 0, double y = 0, double z = 0, const char removal_method = 's') const;

    // =================================================================================================================================

    int    to_int() const;
    long   to_long() const;
    long long to_ll() const;
    size_t to_64() const;
    double to_double() const;
    float  to_float() const;

    // =================================================================================================================================
};

// stand-alone function
template<typename T>
xstring to_xstring(const T& num) {
    return xstring(std::to_string(num)); // TODO: use ss
}


#include "std_xstring.h"
