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

#include "xvector.h"

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
	xvector<xstring> split(const xstring& in_pattern) const;
	xvector<xstring> split(xstring&& in_pattern) const;
	xvector<xstring> split(const char splitter) const;

	//// =================================================================================================================================
	//   match is based on regex_match
	bool match(const xstring& in_pattern) const;
	bool match_line(const xstring& in_pattern) const;
	bool match_lines(const xstring& in_pattern) const;

	//   scan is based on regex_search
	bool scan(const char in_pattern) const;
	bool scan(const xstring& in_pattern) const;
	bool scan_line(const xstring& in_pattern) const;
	bool scan_lines(const xstring& in_pattern) const;

	// exact match (no regex)
	bool is(const xstring& other) const;
	size_t hash() const;

	// =================================================================================================================================

	bool has_non_ascii() const;
	xstring remove_non_ascii() const;

	// =================================================================================================================================
private:
	// re::search & re::findall use grouper/iterator, don't use them via the namespace directly
	xvector<xstring> grouper(const xstring& content, xvector<xstring>& ret_vector, const xstring& in_pattern) const;
	xvector<xstring> iterate(const xstring& content, xvector<xstring>& ret_vector, const xstring& in_pattern) const;
	// --------------------------------------------------------------------------------------------------------------------------------
public:
	std::vector<xstring> findall(const std::string& in_pattern, const bool group = false) const;
	// =================================================================================================================================

	bool has(const char var_char) const;
	bool lacks(const char var_char) const;
	unsigned long long count(const char var_char) const;
	unsigned long long count(const xstring& in_pattern) const;

	// =================================================================================================================================

	xstring sub(const std::string& in_pattern, const std::string& replacement) const;
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
xstring to_xstring(const T& str) {
	return xstring(std::to_string(str)); // TODO: use ss
}


#if defined(__unix__)
	#define   _NODISCARD
	constexpr size_t _FNV_offset_basis = 14695981039346656037ULL;
	constexpr size_t _FNV_prime = 1099511628211ULL;
#endif

// the following is an implementation from Type_Traits
namespace std
{

	_NODISCARD inline size_t _xstring_append_bytes(size_t _Val, const unsigned char* const _First,
		const size_t _Count) noexcept { // accumulate range [_First, _First + _Count) into partial FNV-1a hash _Val
		for (size_t _Idx = 0; _Idx < _Count; ++_Idx) {
			_Val ^= static_cast<size_t>(_First[_Idx]);
			_Val *= _FNV_prime;
		}
		return _Val;
	}

	_NODISCARD inline size_t xstring_Hash_array_representation(
		const char* const _First, size_t _Count) noexcept { // bitwise hashes the representation of an array
		static_assert(is_trivial_v<char*>, "Only trivial types can be directly hashed.");
		return _xstring_append_bytes(
			_FNV_offset_basis, reinterpret_cast<const unsigned char*>(_First), _Count * sizeof(char*));
	}

	template<>
	struct hash<xstring> {
		_NODISCARD size_t operator()(const xstring& str) const
		{ // hash _Keyval to size_t value by pseudorandomizing transform
			return xstring_Hash_array_representation(str.c_str(), str.size());
		}
	};
	template<>
	struct hash<xstring*> {
		_NODISCARD size_t operator()(const xstring* str) const
		{ // hash _Keyval to size_t value by pseudorandomizing transform
			return xstring_Hash_array_representation(str->c_str(), str->size());
		}
	};
}
