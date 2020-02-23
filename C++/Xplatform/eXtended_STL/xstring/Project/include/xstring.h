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
#include<ctype.h>

#include "re2/re2.h"

#include "Color.h"
#include "xvector.h"

namespace rxm {
    using namespace std::regex_constants;
    using type = syntax_option_type;
}
class xstring;
template<typename T>
xstring to_xstring(const T& obj);

class xstring : public std::string
{
    using ull = unsigned long long;
public:
    using std::string::basic_string;

    xstring(std::initializer_list<char> lst) : std::string(std::move(lst)) {};
    xstring(const std::string& str) : std::string(str) {};
    xstring(std::string&& str) noexcept : std::string(std::move(str)) {};
    xstring(size_t repeat, const char chr) : std::string(repeat, chr) {};
    xstring(const char chr);
    xstring(const char* chrs);
    xstring(const unsigned char* chrs);

    // operator=(&&) tests positive for moving in the examples

    void operator+=(const char chr);
    xstring operator+(const char chr);
    void operator+=(const char* chr);
    xstring operator+(const char* chr);
    void operator+=(const unsigned char* chr);
    xstring operator+(const unsigned char* chr);

    void operator+=(const std::string& str);
    xstring operator+(const std::string& str);
    void operator+=(std::string&& str);
    xstring operator+(std::string&& str);

    void print() const;
    void print(int num) const;
    void print(const xstring& front, const xstring& end = "") const;
    void print(const char chr1, const char chr2 = ' ') const;
    void print(const char* chr1, const char* chr2 = "") const;

    std::string to_string() const;

    xstring upper() const;
    xstring lower() const;

    xstring operator*(int total) const;
    void operator*=(int total);

    xstring remove(const char val) const;

    xstring reverse() const;
    // =================================================================================================================================
    
    xvector<xstring> split(size_t loc) const;
    xvector<xstring> split(const std::regex& rex) const;
    xvector<xstring> split(const xstring& pattern, rxm::type mod = rxm::ECMAScript) const;
    xvector<xstring> split(const char splitter, rxm::type mod = rxm::ECMAScript) const;

    xvector<xstring> inclusive_split(const std::regex& rex, bool single = true) const;
    xvector<xstring> inclusive_split(const xstring& splitter, rxm::type mod = rxm::ECMAScript, bool single = true) const;
    xvector<xstring> inclusive_split(const char* splitter, rxm::type mod = rxm::ECMAScript, bool aret = true) const;
    xvector<xstring> inclusive_split(const char splitter, rxm::type mod = rxm::ECMAScript, bool aret = true) const;

    //// =================================================================================================================================
    //   match is based on regex_match
    bool match(const std::regex& rex) const;
    bool match(const re2::RE2& rex) const;
    bool match(const xstring& pattern, rxm::type mod = rxm::ECMAScript) const;
    bool match(const char* str, rxm::type mod = rxm::ECMAScript) const;

    bool match_line(const std::regex& rex) const;
    bool match_line(const re2::RE2& rex) const;
    bool match_line(const xstring& pattern, rxm::type mod = rxm::ECMAScript) const;
    bool match_line(const char* str, rxm::type mod = rxm::ECMAScript) const;

    bool match_lines(const std::regex& rex) const;
    bool match_lines(const re2::RE2& rex) const;
    bool match_lines(const xstring& pattern, rxm::type mod = rxm::ECMAScript) const;
    bool match_lines(const char* str, rxm::type mod = rxm::ECMAScript) const;

    //   scan is based on regex_search
    bool scan(const std::regex& rex) const;
    bool scan(const re2::RE2& rex) const;
    bool scan(const char pattern, rxm::type mod = rxm::ECMAScript) const;
    bool scan(const xstring& pattern, rxm::type mod = rxm::ECMAScript) const;
    bool scan(const char* str, rxm::type mod = rxm::ECMAScript) const;

    bool scan_line(const std::regex& rex) const;
    bool scan_line(const re2::RE2& rex) const;
    bool scan_line(const xstring& pattern, rxm::type mod = rxm::ECMAScript) const;
    bool scan_line(const char* str, rxm::type mod = rxm::ECMAScript) const;

    bool scan_lines(const std::regex& rex) const;
    bool scan_lines(const re2::RE2& rex) const;
    bool scan_lines(const xstring& pattern, rxm::type mod = rxm::ECMAScript) const;
    bool scan_lines(const char* str, rxm::type mod = rxm::ECMAScript) const;

    bool scan_list(const xvector<std::regex>& rex_lst) const;
    bool scan_list(const xvector<re2::RE2>& rex_lst) const;
    bool scan_list(const xvector<xstring>& lst, rxm::type mod = rxm::ECMAScript) const;

    bool scan_list(const xvector<std::regex*>& rex_lst) const;
    bool scan_list(const xvector<re2::RE2*>& rex_lst) const;
    bool scan_list(const xvector<xstring*>& lst, rxm::type mod = rxm::ECMAScript) const;

    // exact match (no regex)
    bool is(const xstring& other) const;
    size_t hash() const;

    // =================================================================================================================================

    int has_non_ascii(int front_skip = 0, int end_skip = 0, int threshold = 0) const;
    bool has_nulls() const;
    bool has_dual_nulls() const;
    xstring remove_non_ascii() const;

    // =================================================================================================================================

    xvector<xstring> findall(const std::regex& rex) const;
    xvector<xstring> findall(const re2::RE2& rex) const;
    xvector<xstring> findall(const xstring& pattern, rxm::type mod = rxm::ECMAScript) const;
    xvector<xstring> findall(const char* pattern, rxm::type mod = rxm::ECMAScript) const;

    xvector<xstring> findwalk(const std::regex& rex) const;
    xvector<xstring> findwalk(const re2::RE2& rex) const;
    xvector<xstring> findwalk(const xstring& pattern, rxm::type mod = rxm::ECMAScript) const;
    xvector<xstring> findwalk(const char* pattern, rxm::type mod = rxm::ECMAScript) const;

    xvector<xstring> search(const std::regex& rex, int count = -1) const;
    //xvector<xstring> search(const re2::RE2& rex, int count = -1) const;
    xvector<xstring> search(const xstring& pattern, rxm::type mod = rxm::ECMAScript, int count = -1) const;
    xvector<xstring> search(const char* pattern, rxm::type mod = rxm::ECMAScript, int count = -1) const;

    // =================================================================================================================================

    bool has(const char var_char) const;
    bool lacks(const char var_char) const;
    size_t count(const char var_char) const;
    size_t count(const xstring& pattern) const;

    // =================================================================================================================================

    xstring sub(const std::regex& rex, const std::string& replacement) const;
    xstring sub(const RE2& rex, const std::string& replacement) const;
    xstring sub(const RE2& rex, const re2::StringPiece& replacement) const;
    xstring sub(const RE2& rex, const char* replacement) const;
    xstring sub(const std::string& pattern, const std::string& replacement, rxm::type mod = rxm::ECMAScript) const;
    xstring sub(const char* pattern, const std::string& replacement, rxm::type mod = rxm::ECMAScript) const;
    xstring sub(const char* pattern, const char* replacement, rxm::type mod = rxm::ECMAScript) const;

    xstring& trim();
    xstring& ltrim();
    xstring& rtrim();
    xstring& trim(const xstring& trim);
    xstring& ltrim(const xstring& trim);
    xstring& rtrim(const xstring& trim);

    // =================================================================================================================================

    xstring operator()(long double x = 0, long double y = 0, long double z = 0, const char removal_method = 's') const;

    // =================================================================================================================================

    int         to_int() const;
    long        to_long() const;
    long long   to_ll() const;
    size_t      to_64() const;
    double      to_double() const;
    float       to_float() const;

    // =================================================================================================================================

    xstring black() const;
    xstring red() const;
    xstring green() const;
    xstring yellow() const;
    xstring blue() const;
    xstring megenta() const;
    xstring cyan() const;
    xstring grey() const;
    xstring white() const;

    xstring on_black() const;
    xstring on_red() const;
    xstring on_green() const;
    xstring on_yellow() const;
    xstring on_blue() const;
    xstring on_megenta() const;
    xstring on_cyan() const;
    xstring on_grey() const;
    xstring on_white() const;

    xstring reset() const;
    xstring bold() const;
    xstring underline() const;
    xstring invert_color() const;
    // =================================================================================================================================
};

// stand-alone function
template<typename T>
xstring to_xstring(const T& obj)
{
    std::ostringstream ostr;
    ostr << obj;
    return xstring(ostr.str().c_str());
}

#include "std_xstring.h"
